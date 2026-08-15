"""FusionRAG 主类: 串起 文档导入(切分/抽取/合并/入库) 与 查询(检索/生成) 两条链路。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Callable, Optional

from .chunking import chunking_by_token_size
from .config import FusionRAGConfig
from .extract import extract_entities
from .llm import LLMService
from .merge import _combine_descriptions, _vote_entity_type, merge_nodes_and_edges
from .prompts import GRAPH_FIELD_SEP
from .query import QueryParam, QueryResult, rag_query
from .rerank import RerankService
from .sessions import SessionStore
from .storage import (
    _atomic_write_json,
    create_graph_storage,
    create_kv_storage,
    create_vector_storage,
)
from .utils import compute_mdhash_id, get_tokenizer

logger = logging.getLogger("fusionrag")


class FusionRAG:
    """用法::

        rag = FusionRAG(FusionRAGConfig.from_env())
        await rag.ainsert("长文本 ...")
        result = await rag.aquery("问题", QueryParam(mode="hybrid"))
    """

    def __init__(self, config: Optional[FusionRAGConfig] = None) -> None:
        self.config = config or FusionRAGConfig.from_env()
        os.makedirs(self.config.working_dir, exist_ok=True)
        wd = self.config.working_dir

        # KV 存储
        self.full_docs = create_kv_storage("full_docs", wd, self.config)            # 文档原文与状态
        self.text_chunks = create_kv_storage("text_chunks", wd, self.config)        # chunk 原文
        self.llm_cache = create_kv_storage("llm_cache", wd, self.config)            # LLM 结果缓存
        self.entity_chunks = create_kv_storage("entity_chunks", wd, self.config)    # 实体->chunk 账本
        self.relation_chunks = create_kv_storage("relation_chunks", wd, self.config)  # 关系->chunk 账本

        # LLM / embedding
        self.llm = LLMService(self.config, cache_kv=self.llm_cache)
        # Rerank (可选, 未配置 RERANK_MODEL 时 available=False, 自动跳过)
        self.rerank = RerankService(self.config)

        # embedding 指纹校验: 换模型/维度会使存量向量失效, 必须先拦截, 防静默数据损坏
        self._check_embedding_fingerprint(wd)

        # 向量存储
        self.chunks_vdb = create_vector_storage(
            "chunks", wd, self.llm.embed, self.config.embedding_dim, self.config
        )
        self.entities_vdb = create_vector_storage(
            "entities", wd, self.llm.embed, self.config.embedding_dim, self.config
        )
        self.relationships_vdb = create_vector_storage(
            "relationships", wd, self.llm.embed, self.config.embedding_dim, self.config
        )

        # 图存储
        self.graph = create_graph_storage("chunk_entity_relation", wd, self.config)

        # 会话管理
        self.sessions = SessionStore(
            wd, self.config.max_history_messages, config=self.config
        )

        self._insert_lock = asyncio.Lock()
        self.tokenizer = get_tokenizer()

    def _check_embedding_fingerprint(self, working_dir: str) -> None:
        """校验 embedding 模型/维度与存量向量是否一致, 防静默数据损坏。

        换 embedding 模型或维度会使已落盘的向量失效(向量空间不兼容),
        若不清库直接复用, SimpleVectorStorage 会把不同维度向量混进同一矩阵,
        导致 np.array 报错或算出错误相似度。这里在启动时用指纹拦截:

        - working_dir/embedding_meta.json 不存在(全新库或旧版遗留) → 写入当前指纹放行;
        - 指纹一致 → 放行;
        - 指纹不一致 → 抛 ValueError, 提示清空 WORKING_DIR 后重新导入
          (llm_cache 可保留复用)。
        """
        path = os.path.join(working_dir, "embedding_meta.json")
        current = {
            "embedding_model": self.config.embedding_model,
            "embedding_dim": self.config.embedding_dim,
        }
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("embedding 指纹文件 %s 损坏, 已按当前配置重写", path)
                saved = None
            if saved is not None:
                if (
                    saved.get("embedding_model") != current["embedding_model"]
                    or saved.get("embedding_dim") != current["embedding_dim"]
                ):
                    raise ValueError(
                        "embedding 配置与存量向量不一致, 拒绝启动以防数据损坏:\n"
                        f"  存量: model={saved.get('embedding_model')!r}, "
                        f"dim={saved.get('embedding_dim')}\n"
                        f"  当前: model={current['embedding_model']!r}, "
                        f"dim={current['embedding_dim']}\n"
                        f"更换 embedding 模型/维度会使已落盘向量失效。请清空 WORKING_DIR "
                        f"({working_dir}) 后重新导入文档 (kv_llm_cache 可保留复用), "
                        f"或改回原 embedding 配置。"
                    )
                return
        # 首次写入指纹 (原子写, 复用 storage 的 _atomic_write_json)
        _atomic_write_json(path, current)

    # ------------------------------------------------------------------ 插入

    async def ainsert(
        self,
        text: str,
        doc_id: Optional[str] = None,
        title: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict:
        """导入一段长文本: 切分 -> 向量化 -> 实体关系抽取 -> 去重合并入库。

        幂等: 相同内容(相同 doc_id)重复导入直接跳过。
        progress_callback(stage, current, total): 进度回调, stage 取值
        chunked / vectorized / extracting / merging / finalizing。
        返回 {doc_id, status, chunks, entities, relations}。
        """
        text = text.strip()
        if not text:
            raise ValueError("文档内容不能为空")
        doc_id = doc_id or compute_mdhash_id(text, prefix="doc-")
        if not title:
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            title = first_line[:40] or doc_id
        notify = progress_callback or (lambda stage, cur, total: None)

        async with self._insert_lock:  # 串行化导入, 简化并发合并冲突
            existing = await self.full_docs.get_by_id(doc_id)
            if existing and existing.get("status") == "PROCESSED":
                logger.info("文档 %s 已存在, 跳过", doc_id)
                return {
                    "doc_id": doc_id,
                    "status": "SKIPPED",
                    "chunks": existing.get("chunks_count", 0),
                    "entities": 0,
                    "relations": 0,
                }

            await self.full_docs.upsert(
                {
                    doc_id: {
                        "content": text,
                        "title": title,
                        "status": "PROCESSING",
                        "content_length": len(text),
                        "created_at": time.time(),
                        "updated_at": time.time(),
                    }
                }
            )

            chunks: dict[str, dict] = {}  # 失败时按已切分数量记录, 供删除清理
            try:
                # 1. 切分 (token 滑窗)
                raw_chunks = chunking_by_token_size(
                    text,
                    self.tokenizer,
                    self.config.chunk_token_size,
                    self.config.chunk_overlap_token_size,
                )
                chunks = {
                    compute_mdhash_id(f"{doc_id}-{c['chunk_order_index']}", prefix="chunk-"): {
                        **c,
                        "full_doc_id": doc_id,
                    }
                    for c in raw_chunks
                }
                notify("chunked", len(chunks), len(chunks))

                # 2. chunk 原文入 KV, 内容入向量库
                await self.text_chunks.upsert(chunks)
                await self.chunks_vdb.upsert(
                    {
                        key: {"content": c["content"], "meta": {"full_doc_id": doc_id}}
                        for key, c in chunks.items()
                    }
                )
                notify("vectorized", len(chunks), len(chunks))

                # 3. 实体/关系抽取 (含 gleaning 补抽), 逐 chunk 汇报进度
                nodes, edges = await extract_entities(
                    chunks, self.llm, self.config, progress_callback=notify
                )

                # 4. 去重合并: 写图谱 + 实体/关系向量库
                stats = await merge_nodes_and_edges(
                    nodes,
                    edges,
                    self.graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.entity_chunks,
                    self.relation_chunks,
                    self.llm,
                    self.config,
                    progress_callback=notify,
                )
                notify("finalizing", 1, 1)

                # 5. 状态落库 + 全部 flush
                await self.full_docs.upsert(
                    {
                        doc_id: {
                            "content": text,
                            "title": title,
                            "status": "PROCESSED",
                            "content_length": len(text),
                            "chunks_count": len(chunks),
                            "entities_count": stats["entities"],
                            "relations_count": stats["relations"],
                            "created_at": time.time(),
                            "updated_at": time.time(),
                        }
                    }
                )
                await self._flush()
                return {
                    "doc_id": doc_id,
                    "status": "PROCESSED",
                    "chunks": len(chunks),
                    "entities": stats["entities"],
                    "relations": stats["relations"],
                }
            except Exception as e:
                # FAILED 记录必须保留 chunks_count: 失败前 text_chunks/chunks_vdb
                # 及部分图数据可能已写入, adelete_by_doc_id 依赖 chunks_count
                # 重算 chunk id 才能清理这些残留 (否则成为孤儿数据)
                await self.full_docs.upsert(
                    {
                        doc_id: {
                            "content": text,
                            "title": title,
                            "status": "FAILED",
                            "error_msg": str(e),
                            "content_length": len(text),
                            # 取本次与历史记录的较大值: 重试在切分前失败时
                            # 不丢上一次已写入的 chunk 数
                            "chunks_count": max(
                                len(chunks), (existing or {}).get("chunks_count", 0)
                            ),
                            "created_at": (existing or {}).get("created_at", time.time()),
                            "updated_at": time.time(),
                        }
                    }
                )
                await self._flush()
                raise

    async def alist_documents(self) -> list[dict]:
        """列出全部已导入文档 (不含原文), 按创建时间倒序。"""
        doc_ids = await self.full_docs.keys()
        records = await self.full_docs.get_by_ids(doc_ids)
        docs = []
        for doc_id, rec in zip(doc_ids, records):
            if not rec:
                continue
            docs.append(
                {
                    "doc_id": doc_id,
                    "title": rec.get("title") or doc_id,
                    "status": rec.get("status", "UNKNOWN"),
                    "chunks": rec.get("chunks_count", 0),
                    "entities": rec.get("entities_count", 0),
                    "relations": rec.get("relations_count", 0),
                    "content_length": rec.get("content_length", 0),
                    "error_msg": rec.get("error_msg"),
                    "created_at": rec.get("created_at", 0),
                }
            )
        docs.sort(key=lambda d: d["created_at"], reverse=True)
        return docs

    # ------------------------------------------------------------------ 删除

    async def adelete_by_doc_id(self, doc_id: str) -> dict:
        """删除一篇文档并保持图谱与向量库的数据一致。

        一致性策略 (实体/关系 chunk 账本裁决, 三分支):
          remaining = 账本chunk_ids − 本doc的chunk_ids
          - remaining 为空  → 真删 (图节点/边 + 实体/关系向量 + 账本行)
          - remaining 变少  → 更新 (账本与 source_id 重写; 关系 weight = len(remaining),
                              因 weight 语义即"贡献 chunk 去重计数";
                              且默认重建描述, 见下)
          - remaining 不变  → 不动
        描述重建 (delete_rebuild_descriptions=True 时):
          对存活的实体/关系, 取其剩余 chunk 原文重新走一遍抽取 (命中 LLM 缓存,
          近乎零成本), 用存活 chunk 的记录重新合并 description/entity_type/keywords
          并更新向量, 被删文档的语义信息随之清除; 抽取未命中时回退为仅重写
          source_id/weight (兜底模式, 保留原描述)。
        LLM 缓存不删 (重导入/删除重建均可复用抽取结果)。
        本操作幂等: 扣减是 no-op, 失败后重发同一请求即可续删。
        """
        async with self._insert_lock:  # 与导入互斥
            record = await self.full_docs.get_by_id(doc_id)
            if not record:
                raise KeyError(f"文档不存在: {doc_id}")
            chunks_count = record.get("chunks_count", 0)
            # chunk key 是确定性的, 按序号重算 (无需存储 chunks_list)
            chunk_ids = [
                compute_mdhash_id(f"{doc_id}-{i}", prefix="chunk-")
                for i in range(chunks_count)
            ]
            chunk_id_set = set(chunk_ids)
            stats = {
                "doc_id": doc_id,
                "status": "DELETED",
                "chunks_deleted": len(chunk_ids),
                "entities_deleted": 0,
                "entities_updated": 0,
                "entities_rebuilt": 0,
                "relations_deleted": 0,
                "relations_updated": 0,
                "relations_rebuilt": 0,
            }
            updated_entities: list[tuple[str, list[str]]] = []   # (name, remaining_chunk_ids)
            updated_relations: list[tuple[str, str, list[str]]] = []  # (src, tgt, remaining)

            # 1. 实体裁决 (候选 = 账本与本文档 chunk 有交集的实体;
            #    本地 JSON 规模全量扫描可接受, 规模化时可换专用的实体→文档索引)
            entity_keys = await self.entity_chunks.keys()
            entity_ledgers = await self.entity_chunks.get_by_ids(entity_keys)
            for name, ledger in zip(entity_keys, entity_ledgers):
                old_ids = (ledger or {}).get("chunk_ids", [])
                if not chunk_id_set.intersection(old_ids):
                    continue
                remaining = [c for c in old_ids if c not in chunk_id_set]
                if not remaining:
                    await self._delete_entity(name, stats)
                else:
                    await self.entity_chunks.upsert({name: {"chunk_ids": remaining}})
                    node = await self.graph.get_node(name)
                    if node:
                        node["source_id"] = GRAPH_FIELD_SEP.join(remaining)
                        await self.graph.upsert_node(name, node)
                    updated_entities.append((name, remaining))
                    stats["entities_updated"] += 1

            # 2. 关系裁决 (在实体之后: 被删实体的残留边已在上面连带清理)
            rel_keys = await self.relation_chunks.keys()
            rel_ledgers = await self.relation_chunks.get_by_ids(rel_keys)
            for key, ledger in zip(rel_keys, rel_ledgers):
                old_ids = (ledger or {}).get("chunk_ids", [])
                if not chunk_id_set.intersection(old_ids):
                    continue
                remaining = [c for c in old_ids if c not in chunk_id_set]
                src, tgt = key.split(GRAPH_FIELD_SEP)
                if not remaining:
                    await self._delete_relation(src, tgt, stats)
                else:
                    await self.relation_chunks.upsert({key: {"chunk_ids": remaining}})
                    edge = await self.graph.get_edge(src, tgt)
                    if edge:
                        edge["source_id"] = GRAPH_FIELD_SEP.join(remaining)
                        edge["weight"] = float(len(remaining))  # 重算计数
                        await self.graph.upsert_edge(src, tgt, edge)
                    updated_relations.append((src, tgt, remaining))
                    stats["relations_updated"] += 1

            # 3. 描述重建: 存活实体/关系用剩余 chunk 重抽(LLM 缓存命中)并重合并
            if self.config.delete_rebuild_descriptions and (
                updated_entities or updated_relations
            ):
                await self._rebuild_surviving_descriptions(
                    updated_entities, updated_relations, stats
                )

            # 4. 删 chunk 原文与向量
            await self.chunks_vdb.delete(chunk_ids)
            await self.text_chunks.delete(chunk_ids)

            # 5. 删文档记录并落盘
            await self.full_docs.delete([doc_id])
            await self._flush()
            logger.info("文档 %s 已删除: %s", doc_id, stats)
            return stats

    async def _delete_entity(self, name: str, stats: dict) -> None:
        """真删实体: 先连带清理残留边, 再删图节点/实体向量/账本。"""
        for s, t in await self.graph.get_node_edges(name):
            await self._delete_relation(s, t, stats)
        await self.graph.remove_node(name)
        await self.entities_vdb.delete([compute_mdhash_id(name, prefix="ent-")])
        await self.entity_chunks.delete([name])
        stats["entities_deleted"] += 1

    async def _delete_relation(self, src: str, tgt: str, stats: dict) -> None:
        """真删关系: 图边 + 关系向量(正反两个 id 都删) + 账本。"""
        await self.graph.remove_edge(src, tgt)
        a, b = sorted((src, tgt))
        await self.relationships_vdb.delete(
            [
                compute_mdhash_id(a + b, prefix="rel-"),
                compute_mdhash_id(b + a, prefix="rel-"),
            ]
        )
        await self.relation_chunks.delete([GRAPH_FIELD_SEP.join([a, b])])
        stats["relations_deleted"] += 1

    async def _rebuild_surviving_descriptions(
        self,
        updated_entities: list[tuple[str, list[str]]],
        updated_relations: list[tuple[str, str, list[str]]],
        stats: dict,
    ) -> None:
        """用剩余 chunk 重抽(LLM 缓存命中)并重建存活实体/关系的描述与向量。

        被删文档贡献的语义碎片随 description 一起清除, 而不是只改 source_id
        留下"语义残留"。抽取结果未覆盖的条目
        回退为仅重写 source_id/weight, 不阻断删除。
        """
        # 收集全部存活 chunk 的原文 (剩余 chunk 属于其他文档, 不在删除范围)
        remaining_ids = sorted(
            {cid for _, ids in updated_entities for cid in ids}
            | {cid for _, _, ids in updated_relations for cid in ids}
        )
        records = await self.text_chunks.get_by_ids(remaining_ids)
        chunks = {
            cid: rec for cid, rec in zip(remaining_ids, records) if rec and rec.get("content")
        }
        if not chunks:
            logger.warning("描述重建: 剩余 chunk 原文缺失, 跳过 (仅重写 source_id/weight)")
            return

        nodes, edges = await extract_entities(chunks, self.llm, self.config)
        # 关系按无向排序键聚合 (抽取产出的 (src, tgt) 顺序与存储键可能相反)
        edge_records: dict[tuple[str, str], list[dict]] = {}
        for (s, t), recs in edges.items():
            edge_records.setdefault(tuple(sorted((s, t))), []).extend(recs)

        entity_payloads: dict[str, dict] = {}
        for name, remaining in updated_entities:
            recs = nodes.get(name, [])
            node = await self.graph.get_node(name)
            if not recs or not node:
                if not recs:
                    logger.warning("描述重建: 实体 %s 重抽未命中, 保留原描述", name)
                continue
            description = await _combine_descriptions(
                "", [r["description"] for r in recs], self.llm, self.config
            )
            node["description"] = description
            node["entity_type"] = _vote_entity_type([r["entity_type"] for r in recs])
            await self.graph.upsert_node(name, node)
            entity_payloads[compute_mdhash_id(name, prefix="ent-")] = {
                "content": f"{name}\n{description}",
                "meta": {"entity_name": name},
            }
            stats["entities_rebuilt"] += 1

        relation_payloads: dict[str, dict] = {}
        for src, tgt, remaining in updated_relations:
            a, b = sorted((src, tgt))
            recs = edge_records.get((a, b), [])
            edge = await self.graph.get_edge(a, b)
            if not recs or not edge:
                if not recs:
                    logger.warning("描述重建: 关系 %s-%s 重抽未命中, 保留原描述", a, b)
                continue
            description = await _combine_descriptions(
                "", [r["description"] for r in recs], self.llm, self.config
            )
            keywords = ",".join(
                sorted(
                    {
                        kw.strip()
                        for r in recs
                        for kw in r["keywords"].split(",")
                        if kw.strip()
                    }
                )
            )
            edge["description"] = description
            edge["keywords"] = keywords or edge.get("keywords", "")
            await self.graph.upsert_edge(a, b, edge)
            relation_payloads[compute_mdhash_id(a + b, prefix="rel-")] = {
                "content": f"{edge['keywords']}\t{a}\n{b}\n{description}",
                "meta": {"src_id": a, "tgt_id": b},
            }
            stats["relations_rebuilt"] += 1

        # 批量更新向量 (内部按 embedding_batch_num 分批)
        if entity_payloads:
            await self.entities_vdb.upsert(entity_payloads)
        if relation_payloads:
            await self.relationships_vdb.upsert(relation_payloads)

    async def _flush(self) -> None:
        await asyncio.gather(
            self.full_docs.index_done_callback(),
            self.text_chunks.index_done_callback(),
            self.entity_chunks.index_done_callback(),
            self.relation_chunks.index_done_callback(),
            self.chunks_vdb.index_done_callback(),
            self.entities_vdb.index_done_callback(),
            self.relationships_vdb.index_done_callback(),
            self.graph.index_done_callback(),
        )

    # ------------------------------------------------------------------ 查询

    async def aquery(self, query: str, param: Optional[QueryParam] = None) -> QueryResult:
        param = param or QueryParam()
        return await rag_query(
            query,
            param,
            self.llm,
            self.config,
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.text_chunks,
            rerank_service=self.rerank,
            full_docs_kv=self.full_docs,
        )

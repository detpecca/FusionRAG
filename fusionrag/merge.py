"""实体/关系的去重与合并。

核心策略:
- 实体按 entity_name 聚合, 关系按排序后的 (src, tgt) 无向聚合
- description: 多片段按 GRAPH_FIELD_SEP(<SEP>) 拼接, 精确去重; 超过 token 阈值
  或片段数过多时调 LLM 摘要压缩
- source_id: 保序去重合并, 上限截断
- weight: 贡献 chunk 去重计数 (每个新 source_id 记 1), 与删除时按剩余
  chunk 数重算的口径一致
- entity_type: 新旧投票取多数
- 实体向量内容 = "name\\ndescription"; 关系向量内容 = "keywords\\tsrc\\ntgt\\ndescription"
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from typing import Optional

from .config import FusionRAGConfig
from .llm import LLMService
from .prompts import GRAPH_FIELD_SEP, PROMPTS
from .storage import BaseGraphStorage, BaseVectorStorage, BaseKVStorage
from .utils import compute_mdhash_id, get_tokenizer, merge_source_ids, sanitize_text

logger = logging.getLogger("fusionrag.merge")


async def merge_nodes_and_edges(
    nodes: dict[str, list[dict]],
    edges: dict[tuple[str, str], list[dict]],
    graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    entity_chunks_kv: BaseKVStorage,
    relation_chunks_kv: BaseKVStorage,
    llm: LLMService,
    config: FusionRAGConfig,
    progress_callback=None,
) -> dict[str, int]:
    """把抽取结果合并进图谱与向量库, 返回统计信息。

    图谱逐实体/逐关系合并; 向量内容先收集, 最后批量写入向量库,
    避免每个实体/关系各触发一次 embedding 调用 (限流友好)。

    执行顺序: 实体合并 -> 补占位端点 -> 关系合并, 三阶段串行。
    实体与关系若并发合并, 关系侧的 "has_node 检查 + 补占位节点" 与
    实体侧的完整节点写入存在 check-then-write 竞态, 空占位可能覆盖
    已合并的 description/entity_type; 阶段内部各 key 唯一, 可安全并发。
    """
    semaphore = asyncio.Semaphore(config.llm_max_async * 2)
    total = len(nodes) + len(edges)
    completed = 0

    def _tick() -> None:
        nonlocal completed
        completed += 1
        if progress_callback:
            progress_callback("merging", completed, total)

    async def _bounded_entity(name: str, records: list[dict]) -> Optional[tuple[str, dict]]:
        async with semaphore:
            result = await _merge_entity(name, records, graph, entity_chunks_kv, llm, config)
            _tick()
            return result

    async def _bounded_relation(key: tuple[str, str], records: list[dict]) -> Optional[tuple[str, dict]]:
        async with semaphore:
            result = await _merge_relation(
                key, records, graph, relation_chunks_kv, llm, config,
            )
            _tick()
            return result

    # 阶段 1: 实体合并 (各实体 key 唯一, 并发安全)
    entity_items = await asyncio.gather(
        *[_bounded_entity(n, r) for n, r in nodes.items()]
    )

    # 阶段 2: 关系端点缺失时补占位节点 (串行, 避免与实体/关系合并竞态)
    await _ensure_edge_endpoints(edges, graph, entity_chunks_kv, config)

    # 阶段 3: 关系合并 (各关系 key 唯一, 且不再写节点, 并发安全)
    relation_items = await asyncio.gather(
        *[_bounded_relation(k, r) for k, r in edges.items()]
    )

    # 批量写向量库 (一次 upsert 内部按 embedding_batch_num 分批)
    await entities_vdb.upsert(dict(item for item in entity_items if item))
    await relationships_vdb.upsert(dict(item for item in relation_items if item))
    return {"entities": len(nodes), "relations": len(edges)}


async def _ensure_edge_endpoints(
    edges: dict[tuple[str, str], list[dict]],
    graph: BaseGraphStorage,
    entity_chunks_kv: BaseKVStorage,
    config: FusionRAGConfig,
) -> None:
    """为缺失的关系端点补 UNKNOWN 占位节点并登记 chunk 账本。

    必须在实体合并全部完成后串行执行: 若与实体合并并发,
    "has_node 为假 -> await -> upsert 占位" 的窗口内实体合并可能已写入
    完整节点, 占位节点的空 description 会将其覆盖 (nx.add_node 更新同名属性)。
    """
    # 聚合各端点在本批全部关系记录中的 source_id
    endpoint_sources: dict[str, list[str]] = {}
    for (src, tgt), records in edges.items():
        for endpoint in (src, tgt):
            endpoint_sources.setdefault(endpoint, []).extend(
                r["source_id"] for r in records
            )

    for endpoint, source_ids in endpoint_sources.items():
        if await graph.has_node(endpoint):
            continue
        await graph.upsert_node(
            endpoint,
            {
                "entity_id": endpoint,
                "entity_type": "unknown",
                "description": "",
                "source_id": merge_source_ids(
                    "", source_ids, GRAPH_FIELD_SEP, config.max_source_ids_per_entity
                ),
                "created_at": time.time(),
            },
        )
        ledger = await entity_chunks_kv.get_by_id(endpoint) or {"chunk_ids": []}
        merged_ids = list(dict.fromkeys(ledger.get("chunk_ids", []) + source_ids))
        await entity_chunks_kv.upsert({endpoint: {"chunk_ids": merged_ids}})


# ---------------------------------------------------------------------------
# description 合并
# ---------------------------------------------------------------------------


def _dedup_descriptions(parts: list[str]) -> list[str]:
    """精确去重, 保序。"""
    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        p = sanitize_text(p)
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result


async def _combine_descriptions(
    existing_description: str, new_descriptions: list[str], llm: LLMService, config: FusionRAGConfig
) -> str:
    """合并新旧描述: <SEP> 拼接 + 去重; 超阈值时 LLM 摘要。"""
    parts = _dedup_descriptions(
        (existing_description.split(GRAPH_FIELD_SEP) if existing_description else [])
        + new_descriptions
    )
    combined = GRAPH_FIELD_SEP.join(parts)
    tokenizer = get_tokenizer()
    need_summary = (
        len(parts) >= config.force_llm_summary_on_merge
        or tokenizer.count(combined) > config.summary_max_tokens
    )
    if not need_summary or not combined:
        return combined
    # LLM 摘要 (单轮压缩)
    description_list = "\n".join(json.dumps({"Description": p}, ensure_ascii=False) for p in parts)
    prompt = PROMPTS["summarize_entity_descriptions"].format(
        summary_length=config.summary_max_tokens, description_list=description_list
    )
    summary = await llm.chat(prompt, use_cache=True, cache_type="summary")
    return sanitize_text(str(summary)) or combined


def _vote_entity_type(candidates: list[str]) -> str:
    counter = Counter(t for t in candidates if t)
    return counter.most_common(1)[0][0] if counter else "unknown"


# ---------------------------------------------------------------------------
# 实体合并
# ---------------------------------------------------------------------------


async def _merge_entity(
    entity_name: str,
    records: list[dict],
    graph: BaseGraphStorage,
    entity_chunks_kv: BaseKVStorage,
    llm: LLMService,
    config: FusionRAGConfig,
) -> tuple[str, dict]:
    """合并实体并返回待写入向量库的 (id, payload)。"""
    existing = await graph.get_node(entity_name)
    ledger = await entity_chunks_kv.get_by_id(entity_name) or {"chunk_ids": []}

    new_descriptions = [r["description"] for r in records]
    new_source_ids = [r["source_id"] for r in records]

    description = await _combine_descriptions(
        existing.get("description", "") if existing else "", new_descriptions, llm, config
    )
    source_id = merge_source_ids(
        existing.get("source_id", "") if existing else "",
        new_source_ids,
        GRAPH_FIELD_SEP,
        config.max_source_ids_per_entity,
    )
    entity_type = _vote_entity_type(
        ([existing["entity_type"]] if existing else [])
        + [r["entity_type"] for r in records]
    )

    node_data = {
        "entity_id": entity_name,
        "entity_type": entity_type,
        "description": description,
        "source_id": source_id,
        "created_at": existing.get("created_at", time.time()) if existing else time.time(),
    }
    await graph.upsert_node(entity_name, node_data)

    # 完整 chunk 账本 (不截断), 供删除/追溯用
    ledger_ids = list(dict.fromkeys(ledger.get("chunk_ids", []) + new_source_ids))
    await entity_chunks_kv.upsert({entity_name: {"chunk_ids": ledger_ids}})

    # 实体向量: embedding 文本 = name + "\n" + description
    return compute_mdhash_id(entity_name, prefix="ent-"), {
        "content": f"{entity_name}\n{description}",
        "meta": {"entity_name": entity_name},
    }


# ---------------------------------------------------------------------------
# 关系合并
# ---------------------------------------------------------------------------


def _relation_chunk_key(src: str, tgt: str) -> str:
    return GRAPH_FIELD_SEP.join(sorted([src, tgt]))


async def _merge_relation(
    edge_key: tuple[str, str],
    records: list[dict],
    graph: BaseGraphStorage,
    relation_chunks_kv: BaseKVStorage,
    llm: LLMService,
    config: FusionRAGConfig,
) -> tuple[str, dict]:
    """合并关系并返回待写入向量库的 (id, payload)。

    端点节点不在此处创建 (见 _ensure_edge_endpoints), 本函数只写边与账本。
    """
    src, tgt = sorted(edge_key)  # 无向: 排序后作为存储键
    existing = await graph.get_edge(src, tgt)
    ledger_key = _relation_chunk_key(src, tgt)
    ledger = await relation_chunks_kv.get_by_id(ledger_key) or {"chunk_ids": []}

    old_source_ids = set(existing["source_id"].split(GRAPH_FIELD_SEP)) if existing else set()

    # weight = 贡献 chunk 去重计数: 每个新 source_id 记 1 (同一 chunk 的多条
    # 记录不重复计), 与删除时按剩余 chunk 数重算 (len(remaining)) 口径一致
    new_unique_ids = {r["source_id"] for r in records} - old_source_ids
    weight = (existing.get("weight", 0.0) if existing else 0.0) + float(len(new_unique_ids))

    # keywords: 新旧 union 后字典序逗号连接
    old_keywords = set(existing["keywords"].split(",")) if existing else set()
    all_keywords = old_keywords | {
        kw.strip() for r in records for kw in r["keywords"].split(",") if kw.strip()
    }
    keywords = ",".join(sorted(all_keywords))

    description = await _combine_descriptions(
        existing.get("description", "") if existing else "",
        [r["description"] for r in records],
        llm,
        config,
    )
    source_id = merge_source_ids(
        existing.get("source_id", "") if existing else "",
        [r["source_id"] for r in records],
        GRAPH_FIELD_SEP,
        config.max_source_ids_per_entity,
    )

    edge_data = {
        "weight": weight,
        "description": description,
        "keywords": keywords,
        "source_id": source_id,
        "created_at": existing.get("created_at", time.time()) if existing else time.time(),
    }

    await graph.upsert_edge(src, tgt, edge_data)

    ledger_ids = list(dict.fromkeys(ledger.get("chunk_ids", []) + [r["source_id"] for r in records]))
    await relation_chunks_kv.upsert({ledger_key: {"chunk_ids": ledger_ids}})

    # 关系向量: embedding 文本 = keywords + "\t" + src + "\n" + tgt + "\n" + description
    return compute_mdhash_id(src + tgt, prefix="rel-"), {
        "content": f"{keywords}\t{src}\n{tgt}\n{description}",
        "meta": {"src_id": src, "tgt_id": tgt},
    }

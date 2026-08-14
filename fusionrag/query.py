"""检索与问答。

检索模式:
- naive : 纯 chunk 向量检索
- local : 低层关键词 -> 实体向量库 top_k -> 一跳关系 + 关联 chunks
- global: 高层关键词 -> 关系向量库 top_k -> 相连实体 + 关联 chunks
- hybrid: local + global round-robin 合并, 外加 chunk 向量检索

检索后处理:
- rerank (可选): 合并后的 chunk 列表过 rerank 模型精排, 低分过滤 + top_n 截断,
  服务不可用时回退原始排序
- 引用溯源: 按最终进入上下文的 chunks 生成 references 列表 (doc_id/title/摘录)

上下文组装: 实体/关系/片段三路分别按 token 预算截断, 拼进 kg_query_context 模板,
交给 rag_response 系统 prompt 生成答案 (支持流式与对话历史)。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import numpy as np

from .config import FusionRAGConfig
from .llm import LLMService
from .prompts import GRAPH_FIELD_SEP, PROMPTS
from .rerank import RerankService, apply_rerank
from .storage import (
    BaseKVStorage,
    BaseGraphStorage,
    BaseVectorStorage,
)
from .utils import cosine_similarity, get_tokenizer, truncate_list_by_token_size

logger = logging.getLogger("fusionrag.query")

QUERY_MODES = ("local", "global", "hybrid", "naive")


@dataclass
class QueryParam:
    mode: str = "hybrid"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 40
    chunk_top_k: int = 20
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 30000
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    user_prompt: str = ""
    enable_rerank: bool = True        # 配置了 rerank 模型时才实际生效
    include_references: bool = True   # raw_data 中附带引用列表


@dataclass
class QueryResult:
    content: Optional[str] = None
    response_iterator: Optional[AsyncIterator[str]] = None
    is_streaming: bool = False
    context: str = ""
    raw_data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------


def _parse_keywords_payload(text: str) -> tuple[list[str], list[str]]:
    """容错解析 LLM 返回的关键词 JSON。"""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return [], []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], []
    hl = payload.get("high_level_keywords") or []
    ll = payload.get("low_level_keywords") or []
    return [str(k) for k in hl], [str(k) for k in ll]


async def extract_keywords(
    query: str, llm: LLMService, config: FusionRAGConfig
) -> tuple[list[str], list[str]]:
    prompt = PROMPTS["keywords_extraction"].format(
        query=query,
        examples=PROMPTS["keywords_extraction_examples"],
        language=config.language,
    )
    result = await llm.chat(prompt, use_cache=True, cache_type="keywords")
    return _parse_keywords_payload(str(result))


# ---------------------------------------------------------------------------
# local / global / naive 三路检索
# ---------------------------------------------------------------------------


async def _local_retrieve(
    ll_keywords: list[str],
    query_embedding: np.ndarray,
    param: QueryParam,
    config: FusionRAGConfig,
    graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_kv: BaseKVStorage,
    chunks_vdb: BaseVectorStorage,
) -> tuple[list[dict], list[dict], list[dict]]:
    """低层关键词 -> 实体 -> 一跳关系 + 关联文本片段。"""
    if not ll_keywords:
        return [], [], []
    results = await entities_vdb.query(
        query=", ".join(ll_keywords), top_k=param.top_k, threshold=config.cosine_threshold
    )
    entity_names = [r["meta"]["entity_name"] for r in results]
    nodes_map = await graph.get_nodes_batch(entity_names)
    # 批量取度数, 单次锁内完成, 避免逐实体抢锁 (N+1 优化)
    degrees = await graph.node_degrees_batch(entity_names)
    entities = []
    for name in entity_names:
        node = nodes_map.get(name)
        if node:
            entities.append({**node, "rank": degrees.get(name, 0)})

    # 一跳关系扩展: 批量取各实体关联边, 单次锁内完成
    edge_pairs: set[tuple[str, str]] = set()
    edges_by_node = await graph.get_nodes_edges_batch(entity_names)
    for pairs in edges_by_node.values():
        edge_pairs.update(pairs)
    edges_map = await graph.get_edges_batch(list(edge_pairs))
    relations = [
        {"src_id": s, "tgt_id": t, **data} for (s, t), data in edges_map.items()
    ]
    relations.sort(key=lambda r: r.get("weight", 0.0), reverse=True)

    # 实体关联文本片段
    chunks = await _chunks_from_sources(
        [e.get("source_id", "") for e in entities],
        query_embedding,
        param,
        config,
        text_chunks_kv,
        chunks_vdb,
        source_type="entity",
    )
    return entities, relations, chunks


async def _global_retrieve(
    hl_keywords: list[str],
    query_embedding: np.ndarray,
    param: QueryParam,
    config: FusionRAGConfig,
    graph: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_kv: BaseKVStorage,
    chunks_vdb: BaseVectorStorage,
    exclude_chunk_ids: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """高层关键词 -> 关系 -> 相连实体 + 关联文本片段。"""
    if not hl_keywords:
        return [], [], []
    results = await relationships_vdb.query(
        query=", ".join(hl_keywords), top_k=param.top_k, threshold=config.cosine_threshold
    )
    pairs = [(r["meta"]["src_id"], r["meta"]["tgt_id"]) for r in results]
    edges_map = await graph.get_edges_batch(pairs)
    relations = [
        {"src_id": s, "tgt_id": t, **data} for (s, t), data in edges_map.items()
    ]

    # 相连实体
    endpoint_names = list({n for p in pairs for n in p})
    nodes_map = await graph.get_nodes_batch(endpoint_names)
    entities = [{**node} for node in nodes_map.values()]

    # 关系关联文本片段 (与实体侧已选片段去重)
    chunks = await _chunks_from_sources(
        [r.get("source_id", "") for r in relations],
        query_embedding,
        param,
        config,
        text_chunks_kv,
        chunks_vdb,
        source_type="relationship",
        exclude_ids=exclude_chunk_ids,
    )
    return entities, relations, chunks


async def _chunks_from_sources(
    source_id_fields: list[str],
    query_embedding: np.ndarray,
    param: QueryParam,
    config: FusionRAGConfig,
    text_chunks_kv: BaseKVStorage,
    chunks_vdb: BaseVectorStorage,
    source_type: str,
    exclude_ids: Optional[set[str]] = None,
) -> list[dict]:
    """从实体/关系的 source_id 收集候选 chunk, 按 query 向量相似度选取。

    选取策略: 候选池按来源频次排序, 再用向量相似度精选。
    """
    exclude_ids = exclude_ids or set()
    # 按出现频次保序去重 (频次高的来源优先)
    counts: dict[str, int] = {}
    for field in source_id_fields:
        for cid in field.split(GRAPH_FIELD_SEP):
            if cid and cid not in exclude_ids:
                counts[cid] = counts.get(cid, 0) + 1
    if not counts:
        return []
    candidate_ids = sorted(counts, key=lambda c: -counts[c])
    max_count = max(config.related_chunk_number, config.related_chunk_number * len(source_id_fields) // 2)
    candidate_ids = candidate_ids[: max_count * 3]  # 候选池, 再按相似度精选

    records = await text_chunks_kv.get_by_ids(candidate_ids)
    valid = [(cid, rec) for cid, rec in zip(candidate_ids, records) if rec]
    if not valid:
        return []

    # 用 chunks_vdb 里已存的向量与 query 计算相似度
    vectors, kept = [], []
    for cid, rec in valid:
        stored = await chunks_vdb.get_by_id(cid)
        if stored is not None:
            vectors.append(stored["vector"])
            kept.append((cid, rec))
    if not kept:
        return []
    sims = cosine_similarity(query_embedding, np.array(vectors, dtype=np.float32))[0]
    order = np.argsort(-sims)[:max_count]
    return [
        {
            "chunk_id": kept[i][0],
            "content": kept[i][1]["content"],
            "full_doc_id": kept[i][1].get("full_doc_id", ""),
            "source_type": source_type,
            "score": float(sims[i]),
        }
        for i in order
    ]


async def _naive_retrieve(
    query: str,
    param: QueryParam,
    config: FusionRAGConfig,
    chunks_vdb: BaseVectorStorage,
) -> list[dict]:
    results = await chunks_vdb.query(
        query=query, top_k=param.chunk_top_k, threshold=config.cosine_threshold
    )
    return [
        {
            "chunk_id": r["id"],
            "content": r["content"],
            "full_doc_id": r.get("meta", {}).get("full_doc_id", ""),
            "source_type": "vector",
            "score": r["distance"],
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# round-robin 合并 (local/global 两路结果交错去重)
# ---------------------------------------------------------------------------


def _round_robin_merge(list_a: list[dict], list_b: list[dict], key) -> list[dict]:
    seen: set = set()
    merged: list[dict] = []
    for item in (x for pair in zip(list_a, list_b) for x in pair):
        k = key(item)
        if k not in seen:
            seen.add(k)
            merged.append(item)
    # 尾部补齐
    for item in list_a[len(list_b):] + list_b[len(list_a):]:
        k = key(item)
        if k not in seen:
            seen.add(k)
            merged.append(item)
    return merged


def _merge_chunk_lists(lists: list[list[dict]]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    max_len = max((len(l) for l in lists), default=0)
    for i in range(max_len):
        for lst in lists:
            if i < len(lst) and lst[i]["chunk_id"] not in seen:
                seen.add(lst[i]["chunk_id"])
                merged.append(lst[i])
    return merged


async def _build_references(
    chunks: list[dict], full_docs_kv: Optional[BaseKVStorage]
) -> list[dict]:
    """按最终进入上下文的 chunks 生成引用列表。

    每个引用对应一个 chunk: 文档 id/标题 + 相关分 + 内容摘录, 供前端展示与答案溯源。
    """
    doc_ids = sorted({c.get("full_doc_id", "") for c in chunks if c.get("full_doc_id")})
    titles: dict[str, str] = {d: d for d in doc_ids}
    if full_docs_kv is not None and doc_ids:
        records = await full_docs_kv.get_by_ids(doc_ids)
        for doc_id, rec in zip(doc_ids, records):
            if rec:
                titles[doc_id] = rec.get("title") or doc_id

    references = []
    seen: set[str] = set()
    for c in chunks:
        if c["chunk_id"] in seen:
            continue
        seen.add(c["chunk_id"])
        doc_id = c.get("full_doc_id", "")
        references.append(
            {
                "reference_id": str(len(references) + 1),
                "chunk_id": c["chunk_id"],
                "doc_id": doc_id,
                "title": titles.get(doc_id, doc_id),
                "score": c.get("rerank_score", c.get("score")),
                "excerpt": c["content"][:120],
            }
        )
    return references


# ---------------------------------------------------------------------------
# 主查询入口
# ---------------------------------------------------------------------------


async def rag_query(
    query: str,
    param: QueryParam,
    llm: LLMService,
    config: FusionRAGConfig,
    graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_kv: BaseKVStorage,
    rerank_service: Optional[RerankService] = None,
    full_docs_kv: Optional[BaseKVStorage] = None,
) -> QueryResult:
    if param.mode not in QUERY_MODES:
        raise ValueError(f"未知查询模式: {param.mode}, 支持 {QUERY_MODES}")

    tokenizer = get_tokenizer()
    query_embedding = (await llm.embed([query]))[0]

    entities: list[dict] = []
    relations: list[dict] = []
    vector_chunks: list[dict] = []
    entity_chunks: list[dict] = []
    relation_chunks: list[dict] = []
    hl_keywords: list[str] = []
    ll_keywords: list[str] = []

    if param.mode == "naive":
        vector_chunks = await _naive_retrieve(query, param, config, chunks_vdb)
    else:
        # 1. 关键词提取 (允许调用方预置)
        hl_keywords = param.hl_keywords
        ll_keywords = param.ll_keywords
        if not hl_keywords and not ll_keywords:
            hl_keywords, ll_keywords = await extract_keywords(query, llm, config)
        if not hl_keywords and not ll_keywords:
            if len(query) < 50:
                ll_keywords = [query]  # 兜底: 短查询整体作为低层关键词
            else:
                return QueryResult(content=PROMPTS["fail_response"], raw_data={"keywords": {}})

        # 2. local / global 检索
        local_entities: list[dict] = []
        local_relations: list[dict] = []
        global_entities: list[dict] = []
        global_relations: list[dict] = []
        if param.mode in ("local", "hybrid"):
            local_entities, local_relations, entity_chunks = await _local_retrieve(
                ll_keywords, query_embedding, param, config, graph,
                entities_vdb, text_chunks_kv, chunks_vdb,
            )
        if param.mode in ("global", "hybrid"):
            exclude = {c["chunk_id"] for c in entity_chunks}
            global_entities, global_relations, relation_chunks = await _global_retrieve(
                hl_keywords, query_embedding, param, config, graph,
                relationships_vdb, text_chunks_kv, chunks_vdb, exclude,
            )
        # round-robin 合并 local/global 结果
        entities = _round_robin_merge(local_entities, global_entities, lambda e: e["entity_id"])
        relations = _round_robin_merge(
            local_relations, global_relations, lambda r: tuple(sorted((r["src_id"], r["tgt_id"])))
        )
        if param.mode == "hybrid":
            vector_chunks = await _naive_retrieve(query, param, config, chunks_vdb)

    # 3. token 预算截断 (实体/关系各自限额)
    entities = truncate_list_by_token_size(
        entities, lambda e: e.get("description", ""), param.max_entity_tokens
    )
    relations = truncate_list_by_token_size(
        relations, lambda r: r.get("description", ""), param.max_relation_tokens
    )

    # 4. 片段合并 + rerank 精排 (可选) + 动态 token 预算
    all_chunks = _merge_chunk_lists([vector_chunks, entity_chunks, relation_chunks])
    if param.enable_rerank:
        all_chunks = await apply_rerank(query, all_chunks, rerank_service, config)
    used_tokens = (
        sum(tokenizer.count(e.get("description", "")) for e in entities)
        + sum(tokenizer.count(r.get("description", "")) for r in relations)
        + tokenizer.count(query)
        + 200  # buffer
    )
    chunk_budget = max(param.max_total_tokens - used_tokens, 0)
    all_chunks = truncate_list_by_token_size(all_chunks, lambda c: c["content"], chunk_budget)

    # 5. 组装上下文
    entities_str = json.dumps(
        [
            {"entity": e["entity_id"], "type": e.get("entity_type", ""), "description": e.get("description", "")}
            for e in entities
        ],
        ensure_ascii=False,
        indent=2,
    )
    relations_str = json.dumps(
        [
            {
                "entity1": r["src_id"],
                "entity2": r["tgt_id"],
                "keywords": r.get("keywords", ""),
                "description": r.get("description", ""),
            }
            for r in relations
        ],
        ensure_ascii=False,
        indent=2,
    )
    chunks_str = json.dumps(
        [{"content": c["content"]} for c in all_chunks], ensure_ascii=False, indent=2
    )
    if param.mode == "naive":
        context = PROMPTS["naive_query_context"].format(text_chunks_str=chunks_str)
    else:
        context = PROMPTS["kg_query_context"].format(
            entities_str=entities_str, relations_str=relations_str, text_chunks_str=chunks_str
        )

    raw_data = {
        "mode": param.mode,
        "keywords": {"high_level": hl_keywords, "low_level": ll_keywords},
        "entities": [e["entity_id"] for e in entities],
        "relations": [(r["src_id"], r["tgt_id"]) for r in relations],
        "chunks": [c["chunk_id"] for c in all_chunks],
        "references": (
            await _build_references(all_chunks, full_docs_kv)
            if param.include_references
            else []
        ),
    }
    if param.only_need_context:
        return QueryResult(content=context, context=context, raw_data=raw_data)

    # 6. 生成答案 (历史只进对话, 不参与检索)
    system_prompt = PROMPTS["rag_response"].format(
        response_type=param.response_type,
        user_prompt=param.user_prompt or "None",
        context_data=context,
    )
    response = await llm.chat(
        query,
        system_prompt=system_prompt,
        history_messages=param.conversation_history,
        stream=param.stream,
        use_cache=True,
        cache_type="query",
    )
    if param.stream:
        return QueryResult(
            response_iterator=response,  # type: ignore[arg-type]
            is_streaming=True,
            context=context,
            raw_data=raw_data,
        )
    return QueryResult(
        content=str(response), context=context, raw_data=raw_data
    )

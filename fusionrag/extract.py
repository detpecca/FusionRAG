"""实体与关系抽取。

流程: 每个 chunk 独立调 LLM 抽取 -> 分隔符格式解析 -> 可选 gleaning 补抽 ->
按实体名 / (src,tgt) 聚合为 {name: [record]} 结构交给 merge 阶段。
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Optional

from .config import FusionRAGConfig
from .llm import LLMService
from .prompts import (
    COMPLETION_DELIMITER,
    DEFAULT_ENTITY_TYPES,
    GRAPH_FIELD_SEP,
    PROMPTS,
    TUPLE_DELIMITER,
)
from .utils import sanitize_text

logger = logging.getLogger("fusionrag.extract")

# 单 chunk 抽取结果: (nodes, edges)
#   nodes: {entity_name: [record, ...]}  record 字段见 _handle_entity
#   edges: {(src, tgt): [record, ...]}   record 字段见 _handle_relation
ChunkExtraction = tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]


async def extract_entities(
    chunks: dict[str, dict],
    llm: LLMService,
    config: FusionRAGConfig,
    progress_callback=None,
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    """对全部 chunk 并发抽取实体/关系并聚合, 逐 chunk 汇报进度。"""
    total = len(chunks)
    completed = 0

    async def _tracked(chunk_key: str, chunk: dict) -> ChunkExtraction:
        nonlocal completed
        result = await _extract_single_chunk(chunk_key, chunk["content"], llm, config)
        completed += 1
        if progress_callback:
            progress_callback("extracting", completed, total)
        return result

    results = await asyncio.gather(*[_tracked(key, chunk) for key, chunk in chunks.items()])

    all_nodes: dict[str, list[dict]] = defaultdict(list)
    all_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for nodes, edges in results:
        for name, records in nodes.items():
            all_nodes[name].extend(records)
        for edge_key, records in edges.items():
            all_edges[edge_key].extend(records)
    return dict(all_nodes), dict(all_edges)


def _build_system_prompt(config: FusionRAGConfig) -> str:
    examples = PROMPTS["entity_extraction_examples"].replace(
        "{tuple_delimiter}", TUPLE_DELIMITER
    ).replace("{completion_delimiter}", COMPLETION_DELIMITER)
    return PROMPTS["entity_extraction_system_prompt"].format(
        entity_types=", ".join(DEFAULT_ENTITY_TYPES),
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        language=config.language,
        max_entity_records=config.max_entity_records,
        max_total_records=config.max_total_records,
        examples=examples,
    )


async def _extract_single_chunk(
    chunk_key: str, content: str, llm: LLMService, config: FusionRAGConfig
) -> ChunkExtraction:
    system_prompt = _build_system_prompt(config)
    user_prompt = PROMPTS["entity_extraction_user_prompt"].format(input_text=content)

    try:
        first_result = await llm.chat(
            user_prompt,
            system_prompt=system_prompt,
            use_cache=True,
            cache_type="extract",
        )
        nodes, edges = _process_extraction_result(str(first_result), chunk_key)

        # gleaning: 把首轮对话作为历史, 让 LLM 补抽遗漏记录
        history = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(first_result)},
        ]
        for _ in range(config.entity_extract_max_gleaning):
            glean_prompt = PROMPTS["entity_continue_extraction_user_prompt"].format(
                tuple_delimiter=TUPLE_DELIMITER,
                completion_delimiter=COMPLETION_DELIMITER,
            )
            glean_result = await llm.chat(
                glean_prompt,
                system_prompt=system_prompt,
                history_messages=history,
                use_cache=True,
                cache_type="extract",
            )
            history.append({"role": "user", "content": glean_prompt})
            history.append({"role": "assistant", "content": str(glean_result)})
            g_nodes, g_edges = _process_extraction_result(str(glean_result), chunk_key)
            nodes, edges = _merge_gleaning(nodes, edges, g_nodes, g_edges)
        return nodes, edges
    except Exception:
        logger.exception("chunk %s 抽取失败, 按空结果处理", chunk_key)
        return {}, {}


def _merge_gleaning(
    nodes: dict[str, list[dict]],
    edges: dict[tuple[str, str], list[dict]],
    g_nodes: dict[str, list[dict]],
    g_edges: dict[tuple[str, str], list[dict]],
) -> ChunkExtraction:
    """合并补抽结果: 同名记录保留 description 更长的版本。"""
    for name, records in g_nodes.items():
        if name not in nodes:
            nodes[name] = records
        else:
            for rec in records:
                old = nodes[name][0]
                if len(rec["description"]) > len(old["description"]):
                    nodes[name][0] = rec
    for key, records in g_edges.items():
        if key not in edges:
            edges[key] = records
        else:
            for rec in records:
                old = edges[key][0]
                if len(rec["description"]) > len(old["description"]):
                    edges[key][0] = rec
    return nodes, edges


def _process_extraction_result(result: str, chunk_key: str) -> ChunkExtraction:
    """解析分隔符格式的抽取输出。"""
    nodes: dict[str, list[dict]] = defaultdict(list)
    edges: dict[tuple[str, str], list[dict]] = defaultdict(list)

    # completion delimiter 之后的内容丢弃
    result = result.split(COMPLETION_DELIMITER)[0]
    for line in result.splitlines():
        line = line.strip()
        if not line or TUPLE_DELIMITER not in line:
            continue
        parts = [p.strip() for p in line.split(TUPLE_DELIMITER)]
        tag = parts[0].lower().strip('"').strip()
        if tag == "entity" and len(parts) >= 4:
            record = _handle_entity(parts[1], parts[2], parts[3], chunk_key)
            if record is not None:
                nodes[record["entity_name"]].append(record)
        elif tag == "relation" and len(parts) >= 5:
            record = _handle_relation(parts[1], parts[2], parts[3], parts[4], chunk_key)
            if record is not None:
                edges[(record["src_id"], record["tgt_id"])].append(record)
    return dict(nodes), dict(edges)


def _handle_entity(
    name: str, entity_type: str, description: str, chunk_key: str
) -> Optional[dict]:
    name = sanitize_text(name)
    # GRAPH_FIELD_SEP 是 source_id / 关系账本 key 的内部分隔符,
    # 实体名混入会破坏账本 key 解析 (删除时 split 解包崩溃), 直接丢弃
    if not name or len(name) > 256 or name.isdigit() or GRAPH_FIELD_SEP in name:
        return None
    # entity_type 规范化: 小写, 逗号分隔取首个
    entity_type = sanitize_text(entity_type).split(",")[0].strip().lower() or "other"
    description = sanitize_text(description)
    return {
        "entity_name": name,
        "entity_type": entity_type,
        "description": description,
        "source_id": chunk_key,
    }


def _handle_relation(
    src: str, tgt: str, keywords: str, description: str, chunk_key: str
) -> Optional[dict]:
    src = sanitize_text(src)
    tgt = sanitize_text(tgt)
    if not src or not tgt or src == tgt:
        return None
    # 端点名同实体名规则: 含 GRAPH_FIELD_SEP 保留字的记录丢弃
    if GRAPH_FIELD_SEP in src or GRAPH_FIELD_SEP in tgt:
        return None
    # 全角逗号转半角
    keywords = sanitize_text(keywords).replace("，", ",")
    description = sanitize_text(description)
    return {
        "src_id": src,
        "tgt_id": tgt,
        "weight": 1.0,
        "description": description,
        "keywords": keywords,
        "source_id": chunk_key,
    }

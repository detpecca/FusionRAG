"""Rerank 精排测试: 重排/低分过滤/top_n 截断/失败兜底/开关与可用性。"""

import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG
from fusionrag.query import QueryParam
from fusionrag.rerank import RerankService, _build_request, _parse_results, apply_rerank

from .conftest import SAMPLE_DOC, fake_embedding

CHUNKS = [
    {"chunk_id": "a", "content": "甲片段", "score": 0.9},
    {"chunk_id": "b", "content": "乙片段", "score": 0.8},
    {"chunk_id": "c", "content": "丙片段", "score": 0.7},
]


def test_build_request_jina_style():
    url, payload = _build_request("https://api.jina.ai/v1", "m", "q", ["d1", "d2"])
    assert url == "https://api.jina.ai/v1/rerank"
    assert payload == {"model": "m", "query": "q", "documents": ["d1", "d2"], "top_n": 2}


def test_build_request_dashscope_style():
    base = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    url, payload = _build_request(base, "gte-rerank-v2", "q", ["d1"])
    assert url == base  # 原生完整服务路径, 不再拼接 /rerank
    assert payload["input"] == {"query": "q", "documents": ["d1"]}
    assert payload["parameters"] == {"top_n": 1, "return_documents": False}


def test_parse_results_both_styles():
    jina = {"results": [{"index": 1, "relevance_score": 0.7}]}
    dashscope = {"output": {"results": [{"index": 0, "relevance_score": 0.4}]}}
    assert _parse_results(jina) == [(1, 0.7)]
    assert _parse_results(dashscope) == [(0, 0.4)]
    assert _parse_results({}) == []


def make_service(tmp_path, rerank_func=None, **overrides):
    config = FusionRAGConfig(
        working_dir=str(tmp_path / "ws"), rerank_func=rerank_func, **overrides
    )
    return RerankService(config), config


async def test_rerank_reorders_and_attaches_score(tmp_path):
    async def reverse_rerank(query, documents):
        return [(2, 0.95), (0, 0.60), (1, 0.40)]

    service, config = make_service(tmp_path, reverse_rerank)
    result = await apply_rerank("q", CHUNKS, service, config)
    assert [c["chunk_id"] for c in result] == ["c", "a", "b"]
    assert result[0]["rerank_score"] == 0.95
    assert result[0]["score"] == 0.7  # 原始向量分保留


async def test_rerank_min_score_and_top_n(tmp_path):
    async def rerank(query, documents):
        return [(0, 0.9), (1, 0.2), (2, 0.8)]

    service, config = make_service(
        tmp_path, rerank, min_rerank_score=0.5, rerank_top_n=1
    )
    result = await apply_rerank("q", CHUNKS, service, config)
    assert [c["chunk_id"] for c in result] == ["a"]  # b 低分丢弃, top_n=1 截断


async def test_rerank_unavailable_or_failure_falls_back(tmp_path):
    # 未配置 rerank (无 func 无 model): 原样返回
    service, config = make_service(tmp_path, None)
    assert not service.available
    assert await apply_rerank("q", CHUNKS, service, config) is CHUNKS
    assert await apply_rerank("q", CHUNKS, None, config) is CHUNKS

    # rerank 调用抛异常: 回退原始排序, 不阻断问答
    async def broken(query, documents):
        raise RuntimeError("rerank service down")

    service, config = make_service(tmp_path, broken)
    assert service.available
    assert await apply_rerank("q", CHUNKS, service, config) is CHUNKS


async def test_rerank_model_config_availability(tmp_path):
    config = FusionRAGConfig(working_dir=str(tmp_path / "ws"), rerank_model="gte-rerank-v2")
    assert RerankService(config).available


async def test_query_pipeline_respects_enable_rerank(tmp_path, fake_llm):
    calls = []

    async def spy_rerank(query, documents):
        calls.append((query, len(documents)))
        return [(i, 1.0 - i * 0.01) for i in range(len(documents))]

    config = FusionRAGConfig(
        working_dir=str(tmp_path / "ws"),
        llm_func=fake_llm,
        embedding_func=fake_embedding,
        rerank_func=spy_rerank,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        cosine_threshold=0.0,
    )
    rag = FusionRAG(config)
    await rag.ainsert(SAMPLE_DOC)

    await rag.aquery("FusionRAG 是什么?", QueryParam(mode="naive", enable_rerank=True))
    assert len(calls) == 1  # naive 检索的 chunks 过了精排

    await rag.aquery("FusionRAG 是什么?", QueryParam(mode="naive", enable_rerank=False))
    assert len(calls) == 1  # 开关关闭, 不再调用

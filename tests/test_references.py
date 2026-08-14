"""引用溯源测试: references 列表结构/标题解析/开关/API 响应。"""

from fastapi.testclient import TestClient

from fusionrag.api import create_app
from fusionrag.core import FusionRAG
from fusionrag.query import QueryParam

from .conftest import SAMPLE_DOC

DOC_TITLE = "FusionRAG 设计说明.md"


async def test_references_structure_and_title(rag):
    await rag.ainsert(SAMPLE_DOC, title=DOC_TITLE)
    result = await rag.aquery("FusionRAG 是什么?", QueryParam(mode="naive"))

    references = result.raw_data["references"]
    assert len(references) == len(result.raw_data["chunks"]) >= 1
    for ref in references:
        assert ref["reference_id"]
        assert ref["chunk_id"] in result.raw_data["chunks"]
        assert ref["doc_id"]
        assert ref["title"] == DOC_TITLE  # 标题从 full_docs 解析
        assert ref["excerpt"]
        assert ref["score"] is not None
    # reference_id 从 1 开始连续编号
    assert [r["reference_id"] for r in references] == [str(i + 1) for i in range(len(references))]


async def test_references_disabled(rag):
    await rag.ainsert(SAMPLE_DOC)
    result = await rag.aquery(
        "FusionRAG 是什么?", QueryParam(mode="naive", include_references=False)
    )
    assert result.raw_data["references"] == []


def test_api_references_and_retrieval(test_config):
    client = TestClient(create_app(FusionRAG(test_config)))
    client.post("/api/v1/document/import", json={"document": SAMPLE_DOC})
    resp = client.post(
        "/api/v1/chat/completions",
        json={"session_id": "ref-1", "message": "FusionRAG 是什么?"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    # references: 结构化引用列表
    assert isinstance(data["references"], list) and len(data["references"]) >= 1
    assert {"reference_id", "chunk_id", "doc_id", "title", "excerpt"} <= set(
        data["references"][0]
    )
    # retrieval: 检索过程数据 (关键词/实体/关系/片段)
    retrieval = data["retrieval"]
    assert "keywords" in retrieval and "entities" in retrieval and "chunks" in retrieval


def test_api_stream_emits_references_frame(test_config):
    client = TestClient(create_app(FusionRAG(test_config)))
    client.post("/api/v1/document/import", json={"document": SAMPLE_DOC})
    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        json={"session_id": "ref-2", "message": "介绍 FusionRAG", "stream": True},
    ) as resp:
        body = "".join(resp.iter_text())
    assert '"references"' in body  # 引用帧在 [DONE] 之前发出
    assert body.index('"references"') < body.index("data: [DONE]")

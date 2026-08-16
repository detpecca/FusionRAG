"""API 测试: 覆盖文档导入/对话/会话历史/会话清理等核心接口 + 文档列表/文件导入。"""

import json

from fastapi.testclient import TestClient

from fusionrag.api import create_app
from fusionrag.core import FusionRAG

from .conftest import SAMPLE_DOC


def _client(test_config) -> TestClient:
    return TestClient(create_app(FusionRAG(test_config)))


def test_document_import(test_config):
    client = _client(test_config)
    resp = client.post("/api/v1/document/import", json={"document": SAMPLE_DOC})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["status"] == "PROCESSED"
    assert data["doc_id"]
    assert data["chunks"] >= 1 and data["entities"] > 0

    # 空文档校验
    resp = client.post("/api/v1/document/import", json={"document": ""})
    assert resp.status_code == 422


def test_chat_and_history_and_delete(test_config):
    client = _client(test_config)
    client.post("/api/v1/document/import", json={"document": SAMPLE_DOC})

    # 第一轮对话 (非流式)
    resp = client.post(
        "/api/v1/chat/completions",
        json={"session_id": "s1", "message": "FusionRAG 是什么?"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["answer"] == "这是基于知识库的回答。"
    assert data["session_id"] == "s1"
    assert "references" in data

    # 第二轮对话 (多轮上下文)
    client.post(
        "/api/v1/chat/completions",
        json={"session_id": "s1", "message": "它和云脑平台什么关系?"},
    )
    resp = client.get("/api/v1/chat/history/s1")
    messages = resp.json()["data"]["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant"

    # 清理会话
    resp = client.delete("/api/v1/chat/session/s1")
    assert resp.json()["data"] == {"session_id": "s1", "cleared": True}
    resp = client.get("/api/v1/chat/history/s1")
    assert resp.json()["data"]["messages"] == []


def test_chat_stream_sse(test_config):
    client = _client(test_config)
    client.post("/api/v1/document/import", json={"document": SAMPLE_DOC})
    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        json={"session_id": "s2", "message": "介绍 FusionRAG", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    assert '"delta"' in body
    assert "data: [DONE]" in body
    # 流式结束后历史已落库
    resp = client.get("/api/v1/chat/history/s2")
    assert len(resp.json()["data"]["messages"]) == 2


def test_chat_invalid_mode(test_config):
    client = _client(test_config)
    resp = client.post(
        "/api/v1/chat/completions",
        json={"session_id": "s3", "message": "hi", "mode": "bogus"},
    )
    assert resp.status_code == 400


def test_webui_served(test_config):
    client = _client(test_config)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "FusionRAG" in resp.text
    assert "/api/v1/chat/completions" in resp.text


def test_documents_list_and_import_file(test_config):
    client = _client(test_config)
    # multipart 文件导入 (SSE 进度流)
    resp = client.post(
        "/api/v1/document/import-file",
        files={"file": ("测试文档.md", SAMPLE_DOC.encode("utf-8"), "text/markdown")},
    )
    assert resp.status_code == 200
    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    stages = [e["stage"] for e in events]
    # 进度事件齐全: 切分 -> 向量化 -> 抽取 -> 合并 -> 收尾 -> 完成
    assert "chunked" in stages and "vectorized" in stages
    assert "extracting" in stages and "merging" in stages
    assert events[-1]["stage"] == "done"
    result = events[-1]["result"]
    assert result["status"] == "PROCESSED" and result["entities"] > 0

    # 文档列表
    resp = client.get("/api/v1/documents")
    data = resp.json()["data"]
    assert data["total"] == 1
    doc = data["documents"][0]
    assert doc["title"] == "测试文档.md"
    assert doc["status"] == "PROCESSED"
    assert doc["entities"] > 0 and doc["chunks"] >= 1


def test_import_file_bad_extension(test_config):
    client = _client(test_config)
    resp = client.post(
        "/api/v1/document/import-file",
        files={"file": ("a.pdf", b"binary", "application/pdf")},
    )
    assert resp.status_code == 400


def test_import_file_empty(test_config):
    client = _client(test_config)
    resp = client.post("/api/v1/document/import-file", data={})
    assert resp.status_code == 400


def test_document_delete(test_config):
    client = _client(test_config)
    resp = client.post("/api/v1/document/import", json={"document": SAMPLE_DOC})
    doc_id = resp.json()["data"]["doc_id"]

    # 删除文档: 返回统计
    resp = client.delete(f"/api/v1/document/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "DELETED"
    assert data["chunks_deleted"] >= 1
    assert data["entities_deleted"] >= 1  # 独占实体被真删

    # 文档列表已不含该文档
    resp = client.get("/api/v1/documents")
    assert resp.json()["data"]["total"] == 0

    # 重复删除: 404
    resp = client.delete(f"/api/v1/document/{doc_id}")
    assert resp.status_code == 404


def test_async_import_and_status_polling(test_config):
    """wait=false 立即返回 PROCESSING, 后台完成, GET 状态端点可轮询到 PROCESSED。

    必须用 context manager 形式的 TestClient: 非 with 用法下每个请求独占一个
    事件循环, 请求结束 loop 即关闭, 后台导入任务会被一并丢弃。
    """
    import time

    with TestClient(create_app(FusionRAG(test_config))) as client:
        resp = client.post(
            "/api/v1/document/import", json={"document": SAMPLE_DOC, "wait": False}
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "PROCESSING" and data["async"] is True
        doc_id = data["doc_id"]

        status = None
        for _ in range(50):  # 最多等 5s
            rec = client.get(f"/api/v1/document/{doc_id}").json()["data"]
            status = rec["status"]
            if status == "PROCESSED":
                break
            time.sleep(0.1)
        assert status == "PROCESSED"
        assert rec["chunks"] >= 1

        # 未知文档 404
        assert client.get("/api/v1/document/doc-not-exist").status_code == 404


def test_stream_disconnect_still_records_history(test_config):
    """流式对话客户端中途断开, 已生成的部分答案仍应落入会话历史。"""
    client = _client(test_config)
    client.post("/api/v1/document/import", json={"document": SAMPLE_DOC})

    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        json={"session_id": "s9", "message": "星尘科技是什么?", "stream": True},
    ) as resp:
        for line in resp.iter_lines():
            if "delta" in line:
                break  # 收到第一个 delta 就断开, 模拟客户端断连

    # 后台生成器 aclose -> finally 落历史
    import time
    for _ in range(30):
        messages = client.get("/api/v1/chat/history/s9").json()["data"]["messages"]
        if messages:
            break
        time.sleep(0.1)
    assert len(messages) == 2
    assert messages[0]["role"] == "user" and messages[0]["content"] == "星尘科技是什么?"
    assert messages[1]["role"] == "assistant" and messages[1]["content"]

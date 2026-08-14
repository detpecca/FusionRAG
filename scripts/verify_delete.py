"""真实删除验证: 导入临时文档 -> 删除 -> 校验一致性统计与文档列表。"""

import json
import sys

import httpx

BASE = "http://localhost:8000"
DOC = (
    "删除验证文档。流式输出是对话接口的核心要求，系统必须以流式形式返回答案。"
    "智能文档助手支持多轮对话，会话历史按滑动窗口管理。"
    "知识库导入接收长文本并拆分为片段，片段经过向量化后持久化存储。"
)

client = httpx.Client(base_url=BASE, timeout=300)

# 1. 导入前文档数
before = client.get("/api/v1/documents").json()["data"]["total"]
print(f"导入前文档数: {before}")

# 2. 导入临时文档
r = client.post("/api/v1/document/import", json={"document": DOC})
data = r.json()["data"]
doc_id = data["doc_id"]
print(f"导入: {json.dumps(data, ensure_ascii=False)}")
assert data["status"] in ("PROCESSED", "SKIPPED"), data
assert client.get("/api/v1/documents").json()["data"]["total"] == before + (1 if data["status"] == "PROCESSED" else 0)

# 3. 删除
r = client.delete(f"/api/v1/document/{doc_id}")
assert r.status_code == 200, r.text
stats = r.json()["data"]
print(f"删除统计: {json.dumps(stats, ensure_ascii=False)}")

# 4. 列表恢复
after = client.get("/api/v1/documents").json()["data"]["total"]
print(f"删除后文档数: {after}")
assert after == before, f"文档数未恢复: {after} != {before}"

# 5. 重复删除 404
r = client.delete(f"/api/v1/document/{doc_id}")
assert r.status_code == 404, r.status_code
print("重复删除返回 404 ✓")

# 6. 已有文档的问答仍正常 (共享数据未被破坏)
r = client.post("/api/v1/chat/completions", json={"session_id": "del-verify", "message": "系统必须实现哪几个接口？"})
answer = r.json()["data"]["answer"]
assert "接口" in answer
print(f"已有知识库问答正常 ✓ (回答前 60 字: {answer[:60]}...)")
client.delete("/api/v1/chat/session/del-verify")
print("\n真实删除验证全部通过")

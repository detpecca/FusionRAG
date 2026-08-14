"""真实端到端测试: 连接真实 LLM + Embedding, 走通完整 RAG 链路。

前置: 服务已在本机 8000 端口启动 (python -m fusionrag 或 docker compose up, 环境变量来自 .env)。
用法: python scripts/e2e_real.py
"""

from __future__ import annotations

import json
import time

import httpx

BASE = "http://localhost:8000"

# 内置示例文档 (虚构公司), 如需测试自己的文档可改为读取本地 .md/.txt 文件
SAMPLE_DOCUMENT = """# 星尘科技：从图增强检索到企业知识库

星尘科技成立于 2019 年, 总部位于杭州, 是一家专注于图增强检索生成 (GraphRAG)
技术的人工智能公司。创始人林岚曾任多家头部互联网公司的搜索架构师。

## 核心产品

云脑平台是星尘科技的旗舰产品, 于 2021 年发布。它使用知识图谱组织实体与关系,
并结合向量检索进行增强问答, 支持文档导入、实体抽取、关系抽取与三路混合检索。
2023 年, 云脑平台引入 rerank 精排与引用溯源, 答案可回溯到具体文档片段。

## 关键事件

- 2019 年 5 月, 林岚与两位合伙人创立星尘科技, 获天使轮融资 1200 万元。
- 2021 年 9 月, 云脑平台 1.0 发布, 首批客户来自金融与政务行业。
- 2022 年, 公司因一起数据合规争议被处以罚款, 随后建立数据治理委员会。
- 2023 年 12 月, 原 CTO 周明离职, 加入智源人工智能研究院; 苏晴接任 CTO。
- 2024 年 6 月, 星尘科技发布云脑平台 3.0, 支持时态知识图谱与知识治理工作流。
"""


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=600)

    section("1. 导入文档 POST /api/v1/document/import")
    t0 = time.time()
    r = client.post("/api/v1/document/import", json={"document": SAMPLE_DOCUMENT})
    r.raise_for_status()
    print(json.dumps(r.json()["data"], ensure_ascii=False, indent=2))
    print(f"耗时 {time.time() - t0:.1f}s")

    section("2. hybrid 问答(非流式) POST /api/v1/chat/completions")
    q1 = "云脑平台是什么？它经历了哪些重要版本迭代？"
    t0 = time.time()
    r = client.post(
        "/api/v1/chat/completions",
        json={"session_id": "real-s1", "message": q1, "mode": "hybrid"},
    )
    data = r.json()["data"]
    print("Q:", q1)
    print("A:", data["answer"])
    print("\n[检索溯源]")
    retrieval = data["retrieval"]
    print("  keywords:", retrieval["keywords"])
    print("  entities:", retrieval["entities"][:10])
    print("  relations:", retrieval["relations"][:5])
    print("  chunks:", len(retrieval["chunks"]), "个片段")
    print("  references:")
    for ref in data["references"][:5]:
        print(f"    [{ref['reference_id']}] {ref['title']} :: {ref['excerpt'][:40]}...")
    print(f"耗时 {time.time() - t0:.1f}s")

    section("3. 多轮对话第二问(自动携带历史)")
    q2 = "它的首任 CTO 后来去了哪里？"
    t0 = time.time()
    r = client.post(
        "/api/v1/chat/completions",
        json={"session_id": "real-s1", "message": q2},
    )
    print("Q:", q2)
    print("A:", r.json()["data"]["answer"])
    print(f"耗时 {time.time() - t0:.1f}s")

    section("4. naive 模式对照(纯向量检索)")
    q3 = "公司经历过哪些争议？"
    r = client.post(
        "/api/v1/chat/completions",
        json={"session_id": "real-s2", "message": q3, "mode": "naive"},
    )
    print("Q:", q3)
    print("A:", r.json()["data"]["answer"])

    section("5. 流式问答 stream=true (SSE)")
    q4 = "星尘科技是如何解决数据合规问题的？"
    print("Q:", q4)
    print("A: ", end="", flush=True)
    t0 = time.time()
    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        json={"session_id": "real-s1", "message": q4, "stream": True},
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            if "delta" in payload:
                print(payload["delta"], end="", flush=True)
            elif "error" in payload:
                print("\n[ERROR]", payload["error"])
    print(f"\n(流式完成, 耗时 {time.time() - t0:.1f}s)")

    section("6. 查看会话历史 GET /api/v1/chat/history/real-s1")
    r = client.get("/api/v1/chat/history/real-s1")
    msgs = r.json()["data"]["messages"]
    print(f"共 {len(msgs)} 条消息:")
    for m in msgs:
        print(f"  [{m['role']:9s}] {m['content'][:70].replace(chr(10), ' ')}...")

    section("7. 清理会话 DELETE /api/v1/chat/session/real-s1")
    r = client.delete("/api/v1/chat/session/real-s1")
    print("清理结果:", r.json()["data"])
    r = client.get("/api/v1/chat/history/real-s1")
    print("清理后历史条数:", len(r.json()["data"]["messages"]))

    print("\n全部真实链路测试完成。")


if __name__ == "__main__":
    main()

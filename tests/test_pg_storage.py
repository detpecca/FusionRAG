"""PostgreSQL 全家桶后端测试: KV (postgres) / 向量 (pgvector) / 图 (AGE)。

需要真实 PG 才能跑 (无 PG 环境自动跳过):
    PG_TEST_DSN      指向带 vector 扩展的 PG  (KV + 向量测试)
    AGE_TEST_DSN     指向带 Apache AGE 扩展的 PG (图测试; 省略时回退 PG_TEST_DSN)

本地快速起测试实例:
    docker run -d -e POSTGRES_PASSWORD=postgres -p 15432:5432 pgvector/pgvector:pg16
    docker run -d -e POSTGRES_PASSWORD=postgres -p 15433:5432 apache/age:latest
    PG_TEST_DSN=postgresql://postgres:postgres@127.0.0.1:15432/postgres \
    AGE_TEST_DSN=postgresql://postgres:postgres@127.0.0.1:15433/postgres pytest tests/test_pg_storage.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG
from fusionrag.query import QueryParam
from fusionrag.storage_pg import (
    AGEGraphStorage,
    PgvectorStorage,
    PostgresKVStorage,
    close_all_pools,
)

from .conftest import SAMPLE_DOC, fake_embedding

VEC_DSN = os.environ.get("PG_TEST_DSN", "")
AGE_DSN = os.environ.get("AGE_TEST_DSN") or VEC_DSN

needs_vec = pytest.mark.skipif(not VEC_DSN, reason="需要环境变量 PG_TEST_DSN (pgvector 实例)")
needs_age = pytest.mark.skipif(not AGE_DSN, reason="需要 AGE 测试实例 (AGE_TEST_DSN 或 PG_TEST_DSN)")


@pytest.fixture()
def ns() -> str:
    """每个测试独立命名空间, 避免共享库中的表互相污染。"""
    return "t" + uuid.uuid4().hex[:8]


@pytest.fixture(autouse=True)
async def _close_pools():
    yield
    await close_all_pools()


# ---------------------------------------------------------------------------
# KV: PostgresKVStorage
# ---------------------------------------------------------------------------


@needs_vec
async def test_kv_upsert_get_keys(ns):
    kv = PostgresKVStorage(ns, VEC_DSN)
    assert await kv.is_empty()
    await kv.upsert({"a": {"x": 1}, "b": {"y": [1, 2, 3], "标题": "中文"}})
    assert not await kv.is_empty()
    assert await kv.get_by_id("a") == {"x": 1}
    assert await kv.get_by_ids(["b", "missing"]) == [{"y": [1, 2, 3], "标题": "中文"}, None]
    assert set(await kv.keys()) == {"a", "b"}
    assert await kv.filter_keys({"a", "z"}) == {"z"}


@needs_vec
async def test_kv_upsert_overwrite_and_delete(ns):
    kv = PostgresKVStorage(ns, VEC_DSN)
    await kv.upsert({"a": {"x": 1}})
    await kv.upsert({"a": {"x": 2}})  # 覆盖
    assert await kv.get_by_id("a") == {"x": 2}
    await kv.delete(["a"])
    assert await kv.get_by_id("a") is None
    assert await kv.is_empty()


@needs_vec
async def test_kv_new_instance_sees_data(ns):
    """PG 持久化即时生效: 新实例 (无 flush) 直接可见。"""
    kv1 = PostgresKVStorage(ns, VEC_DSN)
    await kv1.upsert({"doc-1": {"nested": {"deep": {"v": True}}}})
    kv2 = PostgresKVStorage(ns, VEC_DSN)
    assert await kv2.get_by_id("doc-1") == {"nested": {"deep": {"v": True}}}
    await kv2.index_done_callback()  # no-op 也不应报错


# ---------------------------------------------------------------------------
# 向量: PgvectorStorage
# ---------------------------------------------------------------------------


@needs_vec
async def test_vector_upsert_query_threshold(ns):
    vdb = PgvectorStorage(ns, VEC_DSN, fake_embedding, 128)
    await vdb.upsert(
        {
            "c1": {"content": "星尘科技是一家人工智能公司", "meta": {"full_doc_id": "d1"}},
            "c2": {"content": "云脑平台使用知识图谱", "meta": {"full_doc_id": "d1"}},
        }
    )
    results = await vdb.query(query="星尘科技", top_k=2)
    assert [r["id"] for r in results][0] == "c1"  # 相似度降序, 精确匹配排第一
    assert set(r["id"] for r in results) == {"c1", "c2"}  # top_k=2 返回两条

    # 与 numpy 直算的余弦相似度一致 (保证与 JSON 后端口径统一)
    import numpy as np
    from fusionrag.storage import cosine_similarity
    q_vec = (await fake_embedding(["星尘科技"]))[0]
    c1_vec = (await fake_embedding(["星尘科技是一家人工智能公司"]))[0]
    expected = float(cosine_similarity(q_vec, c1_vec)[0][0])
    assert results[0]["distance"] == pytest.approx(expected, abs=1e-5)
    assert results[0]["meta"] == {"full_doc_id": "d1"}

    # 阈值过滤: 高阈值排除不相似结果 (c2 与查询无共同词, 相似度~0)
    strict = await vdb.query(query="星尘科技", top_k=2, threshold=0.3)
    assert [r["id"] for r in strict] == ["c1"]

    # query_embedding 直查 (复用已算好的向量, 不再 embed)
    emb = (await fake_embedding(["云脑平台"]))[0]
    by_emb = await vdb.query(query_embedding=emb, top_k=1)
    assert by_emb[0]["id"] == "c2"


@needs_vec
async def test_vector_upsert_overwrite_and_delete(ns):
    vdb = PgvectorStorage(ns, VEC_DSN, fake_embedding, 128)
    await vdb.upsert({"c1": {"content": "旧内容", "meta": {}}})
    await vdb.upsert({"c1": {"content": "新内容 星辰大海", "meta": {"v": 2}}})
    got = await vdb.get_by_id("c1")
    assert got is not None
    assert got["content"] == "新内容 星辰大海"
    assert got["meta"] == {"v": 2}
    assert len(got["vector"]) == 128  # query.py 相似度精选依赖 vector 字段
    await vdb.delete(["c1"])
    assert await vdb.get_by_id("c1") is None


@needs_vec
async def test_vector_get_by_ids_batch(ns):
    from fusionrag.storage import BaseVectorStorage  # noqa: F401

    vdb = PgvectorStorage(ns, VEC_DSN, fake_embedding, 128)
    await vdb.upsert({"a": {"content": "x", "meta": {}}, "b": {"content": "y", "meta": {}}})
    got = await vdb.get_by_ids(["b", "missing", "a"])
    assert [g is not None for g in got] == [True, False, True]


# ---------------------------------------------------------------------------
# 图: AGEGraphStorage (语义与 NetworkXGraphStorage 对齐)
# ---------------------------------------------------------------------------


@needs_age
async def test_graph_node_crud(ns):
    g = AGEGraphStorage(ns, AGE_DSN)
    assert not await g.has_node("星尘科技")
    await g.upsert_node(
        "星尘科技",
        {"entity_type": "organization", "description": "AI 公司", "weight": 1.0,
         "source_id": "c1<SEP>c2", "created_at": 1755000000.5},
    )
    assert await g.has_node("星尘科技")
    node = await g.get_node("星尘科技")
    assert node["entity_type"] == "organization"
    assert node["description"] == "AI 公司"
    assert node["source_id"] == "c1<SEP>c2"  # <SEP> 拼接的 source_id 原样保留
    assert node["created_at"] == 1755000000.5
    assert node["weight"] == 1.0

    # upsert 覆盖属性
    await g.upsert_node("星尘科技", {"entity_type": "company", "description": "AI 公司",
                                     "weight": 2.0, "source_id": "c1", "created_at": 1.0})
    assert (await g.get_node("星尘科技"))["weight"] == 2.0

    # 批量读 + 不存在的节点
    batch = await g.get_nodes_batch(["星尘科技", "不存在"])
    assert set(batch) == {"星尘科技"}


@needs_age
async def test_graph_edge_crud_and_undirected(ns):
    g = AGEGraphStorage(ns, AGE_DSN)
    await g.upsert_node("A", {"entity_type": "other", "description": "", "source_id": "c1", "created_at": 1.0})
    await g.upsert_node("B", {"entity_type": "other", "description": "", "source_id": "c1", "created_at": 1.0})
    await g.upsert_edge("B", "A", {"weight": 2.0, "keywords": "k1,k2",
                                   "description": "关系", "source_id": "c1", "created_at": 1.0})

    # 无向: 任意顺序的端点都能命中
    assert await g.has_edge("A", "B")
    assert await g.has_edge("B", "A")
    edge = await g.get_edge("B", "A")
    assert edge["weight"] == 2.0
    assert edge["keywords"] == "k1,k2"

    # 批量取边: key 为排序后的端点对
    got = await g.get_edges_batch([("A", "B"), ("A", "zzz")])
    assert set(got) == {("A", "B")}
    assert got[("A", "B")]["weight"] == 2.0

    # upsert_edge 幂等: 重复执行不重复建边
    await g.upsert_edge("A", "B", {"weight": 3.0, "keywords": "k1",
                                   "description": "关系", "source_id": "c1", "created_at": 1.0})
    deg = await g.node_degree("A")
    assert deg == 1
    assert (await g.get_edge("A", "B"))["weight"] == 3.0


@needs_age
async def test_graph_degrees_and_node_edges(ns):
    g = AGEGraphStorage(ns, AGE_DSN)
    for n in ("A", "B", "C"):
        await g.upsert_node(n, {"entity_type": "other", "description": "",
                                "source_id": "c1", "created_at": 1.0})
    for s, t in (("A", "B"), ("A", "C")):
        await g.upsert_edge(s, t, {"weight": 1.0, "keywords": "", "description": "",
                                   "source_id": "c1", "created_at": 1.0})

    assert await g.node_degree("A") == 2
    assert await g.node_degree("B") == 1
    assert await g.node_degree("zzz") == 0

    degs = await g.node_degrees_batch(["A", "B", "zzz"])
    assert degs == {"A": 2, "B": 1, "zzz": 0}

    edges = await g.get_node_edges("A")
    assert edges == [("A", "B"), ("A", "C")]

    batch = await g.get_nodes_edges_batch(["A", "B"])
    assert batch["A"] == [("A", "B"), ("A", "C")]
    assert batch["B"] == [("A", "B")]


@needs_age
async def test_graph_remove_node_and_edge(ns):
    g = AGEGraphStorage(ns, AGE_DSN)
    for n in ("A", "B", "C"):
        await g.upsert_node(n, {"entity_type": "other", "description": "",
                                "source_id": "c1", "created_at": 1.0})
    for s, t in (("A", "B"), ("B", "C")):
        await g.upsert_edge(s, t, {"weight": 1.0, "keywords": "", "description": "",
                                   "source_id": "c1", "created_at": 1.0})

    await g.remove_edge("C", "B")  # 乱序端点删除
    assert not await g.has_edge("B", "C")
    assert await g.has_node("B")

    await g.remove_node("B")  # DETACH DELETE: 连带清边
    assert not await g.has_node("B")
    assert not await g.has_edge("A", "B")
    assert await g.has_node("A")


# ---------------------------------------------------------------------------
# 端到端: 全家桶配置跑完整 导入/查询/删除/幂等 链路
# ---------------------------------------------------------------------------


@pytest.fixture()
def pg_config(tmp_path, fake_llm):
    """storage_backend=postgres 全家桶; 图单独指向 AGE 实例 (也验证按类 DSN 覆盖)。"""
    if not (VEC_DSN and AGE_DSN):
        pytest.skip("需要 PG_TEST_DSN 与 AGE 实例")
    return FusionRAGConfig(
        working_dir=str(tmp_path / "workspace"),
        storage_backend="postgres",
        postgres_dsn=VEC_DSN,
        pg_dsn_graph=AGE_DSN,
        llm_func=fake_llm,
        embedding_func=fake_embedding,
        embedding_dim=128,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        summary_max_tokens=2000,
        force_llm_summary_on_merge=99,
        cosine_threshold=0.0,
    )


async def _reset_pg_state(cfg: FusionRAGConfig) -> None:
    """e2e 前清场: 共享测试库中可能残留此前 (失败) 运行的数据。

    账本残留会让"删除后实体保留"的断言失真 —— 那是删除语义的正确行为,
    但污染了测试前提, 所以每个 e2e 用例从空库开始。
    """
    import asyncpg

    conn = await asyncpg.connect(cfg.postgres_dsn)
    for t in ("kv_full_docs", "kv_text_chunks", "kv_llm_cache",
              "kv_entity_chunks", "kv_relation_chunks", "kv_chat_sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS public.{t}")
    for t in ("vdb_chunks", "vdb_entities", "vdb_relationships"):
        await conn.execute(f"DROP TABLE IF EXISTS public.{t}")
        await conn.execute(f"DROP INDEX IF EXISTS public.idx_{t[4:]}_hnsw")
    await conn.close()

    age_conn = await asyncpg.connect(cfg.pg_dsn_graph)
    await age_conn.execute("LOAD 'age'")
    n = await age_conn.fetchval(
        "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = $1",
        "fusionrag_chunk_entity_relation",
    )
    if n:
        await age_conn.execute(
            "SELECT ag_catalog.drop_graph($1, true)", "fusionrag_chunk_entity_relation"
        )
    await age_conn.close()
    # 清掉进程内连接池, 避免复用持有旧 prepared statement 的连接
    await close_all_pools()


@needs_vec
async def test_e2e_full_pg_stack_insert_query_delete(pg_config):
    await _reset_pg_state(pg_config)
    doc_id = "e2e-" + uuid.uuid4().hex[:8]
    rag = FusionRAG(pg_config)
    result = await rag.ainsert(SAMPLE_DOC, doc_id=doc_id)
    assert result["status"] == "PROCESSED"
    assert result["entities"] > 0

    # 图 (AGE) 与 KV (postgres 表) 都有数据
    assert await rag.graph.has_node("星尘科技")
    doc = await rag.full_docs.get_by_id(doc_id)
    assert doc is not None and doc["status"] == "PROCESSED"

    # 查询三模式
    for mode in ("naive", "local", "global", "hybrid"):
        res = await rag.aquery("星尘科技是什么?", QueryParam(mode=mode))
        assert res.raw_data["chunks"], mode

    # 幂等重导
    again = await rag.ainsert(SAMPLE_DOC, doc_id=doc_id)
    assert again["status"] == "SKIPPED"

    # 删除: 图/账本/KV 全链路清理
    stats = await rag.adelete_by_doc_id(doc_id)
    assert stats["status"] == "DELETED"
    assert not await rag.graph.has_node("星尘科技")
    assert await rag.full_docs.get_by_id(doc_id) is None
    assert await rag.text_chunks.is_empty()


@needs_vec
async def test_e2e_two_instances_share_pg_state(pg_config):
    """两个实例 (模拟多 worker) 指向同一 PG: 写入互相可见, 无单实例锁拦截。"""
    await _reset_pg_state(pg_config)
    doc_id = "e2e2-" + uuid.uuid4().hex[:8]
    rag1 = FusionRAG(pg_config)
    rag2 = FusionRAG(pg_config)  # 同 working_dir; PG 后端不应触发 instance.lock
    assert not await rag2.graph.has_node("星尘科技")  # 起点无该实体

    await rag1.ainsert(SAMPLE_DOC, doc_id=doc_id)
    # rag2 (未参与导入) 立即可见 —— JSON 后端做不到这一点
    assert await rag2.graph.has_node("星尘科技")
    rec = await rag2.full_docs.get_by_id(doc_id)
    assert rec["status"] == "PROCESSED"
    res = await rag2.aquery("云脑平台是什么?", QueryParam(mode="hybrid"))
    assert res.raw_data["chunks"]

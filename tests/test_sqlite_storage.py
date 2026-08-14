"""SqliteKVStorage 单元测试 + sqlite KV 后端端到端持久化/配置优先级测试。"""

from __future__ import annotations

import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG
from fusionrag.query import QueryParam
from fusionrag.storage import create_kv_storage
from fusionrag.storage_sqlite import SqliteKVStorage

from .conftest import SAMPLE_DOC, fake_embedding


# ---------------------------------------------------------------------------
# 单元测试: 直接操作 SqliteKVStorage
# ---------------------------------------------------------------------------


async def test_upsert_get_and_keys(tmp_path):
    kv = SqliteKVStorage("t", str(tmp_path))
    assert await kv.is_empty()
    await kv.upsert({"a": {"x": 1}, "b": {"y": [1, 2, 3]}})
    assert not await kv.is_empty()
    assert await kv.get_by_id("a") == {"x": 1}
    assert await kv.get_by_ids(["b", "missing"]) == [{"y": [1, 2, 3]}, None]
    assert set(await kv.keys()) == {"a", "b"}
    assert await kv.filter_keys({"a", "z"}) == {"z"}


async def test_delete_and_filter(tmp_path):
    kv = SqliteKVStorage("t", str(tmp_path))
    await kv.upsert({"a": {"x": 1}, "b": {"x": 2}})
    await kv.delete(["a"])
    assert await kv.get_by_id("a") is None
    assert await kv.filter_keys({"a", "b"}) == {"a"}


async def test_incremental_persistence_reload(tmp_path):
    """落盘后新实例应完整重载 (含嵌套 dict / 中文 / list)。"""
    kv1 = SqliteKVStorage("docs", str(tmp_path))
    await kv1.upsert(
        {
            "doc-1": {"title": "标题", "tags": ["中文", "list"], "n": 42},
            "doc-2": {"nested": {"deep": {"v": True}}},
        }
    )
    await kv1.index_done_callback()

    kv2 = SqliteKVStorage("docs", str(tmp_path))  # 模拟重启
    assert await kv2.get_by_id("doc-1") == {"title": "标题", "tags": ["中文", "list"], "n": 42}
    assert await kv2.get_by_id("doc-2") == {"nested": {"deep": {"v": True}}}


async def test_delete_persists_after_flush(tmp_path):
    kv1 = SqliteKVStorage("d", str(tmp_path))
    await kv1.upsert({"a": {"x": 1}, "b": {"x": 2}})
    await kv1.index_done_callback()
    await kv1.delete(["a"])
    await kv1.index_done_callback()

    kv2 = SqliteKVStorage("d", str(tmp_path))
    assert await kv2.get_by_id("a") is None
    assert await kv2.get_by_id("b") == {"x": 2}


async def test_mutate_upsert_reference_pattern(tmp_path):
    """复现 sessions.append_turn: 取出 -> mutate -> upsert -> flush -> 重载。"""
    kv1 = SqliteKVStorage("s", str(tmp_path))
    await kv1.upsert({"s1": {"messages": []}})
    record = await kv1.get_by_id("s1")
    record["messages"].append({"role": "user", "content": "hi"})
    await kv1.upsert({"s1": record})
    await kv1.index_done_callback()

    kv2 = SqliteKVStorage("s", str(tmp_path))
    assert await kv2.get_by_id("s1") == {"messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# 工厂 / 配置优先级
# ---------------------------------------------------------------------------


def test_factory_kv_backend_dispatch(tmp_path):
    cfg = FusionRAGConfig(working_dir=str(tmp_path), kv_backend="sqlite")
    store = create_kv_storage("x", str(tmp_path), cfg)
    assert isinstance(store, SqliteKVStorage)


def test_storage_backend_fallback(tmp_path):
    """未设 kv_backend 时回退到 storage_backend。"""
    cfg = FusionRAGConfig(working_dir=str(tmp_path), storage_backend="sqlite")
    assert cfg.resolve_kv_backend() == "sqlite"
    store = create_kv_storage("x", str(tmp_path), cfg)
    assert isinstance(store, SqliteKVStorage)


def test_kv_backend_overrides_storage_backend(tmp_path):
    cfg = FusionRAGConfig(
        working_dir=str(tmp_path), storage_backend="json", kv_backend="sqlite"
    )
    assert cfg.resolve_kv_backend() == "sqlite"
    assert cfg.resolve_vector_backend() == "json"


def test_unknown_kv_backend_raises(tmp_path):
    cfg = FusionRAGConfig(working_dir=str(tmp_path), kv_backend="foobar")
    with pytest.raises(ValueError, match="KV_BACKEND"):
        create_kv_storage("x", str(tmp_path), cfg)


# ---------------------------------------------------------------------------
# 端到端: sqlite KV 后端下完整导入/重启/会话 (参照 test_persistence.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_config(tmp_path, fake_llm):
    return FusionRAGConfig(
        working_dir=str(tmp_path / "workspace"),
        kv_backend="sqlite",
        llm_func=fake_llm,
        embedding_func=fake_embedding,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        summary_max_tokens=2000,
        force_llm_summary_on_merge=99,
        cosine_threshold=0.0,
    )


async def test_e2e_persistence_with_sqlite_kv(sqlite_config):
    rag1 = FusionRAG(sqlite_config)
    result = await rag1.ainsert(SAMPLE_DOC)
    assert result["status"] == "PROCESSED"
    await rag1.sessions.append_turn("s1", "问题一", "回答一")

    # 同 working_dir 新实例, 模拟重启
    rag2 = FusionRAG(sqlite_config)

    # 图/向量 (仍走 json 后端) 正常
    node = await rag2.graph.get_node("星尘科技")
    assert node is not None
    res = await rag2.aquery("星尘科技是什么?", QueryParam(mode="naive"))
    assert res.raw_data["chunks"]

    # KV (sqlite 后端): 文档幂等 + 会话历史保留
    again = await rag2.ainsert(SAMPLE_DOC)
    assert again["status"] == "SKIPPED"
    history = await rag2.sessions.get_history("s1")
    assert history == [
        {"role": "user", "content": "问题一"},
        {"role": "assistant", "content": "回答一"},
    ]

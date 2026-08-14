"""文档删除与数据一致性测试: 账本裁决三分支 (真删/更新/不动)、weight 重算、重导入无污染。"""

import numpy as np
import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG
from fusionrag.prompts import GRAPH_FIELD_SEP
from fusionrag.utils import compute_mdhash_id

from .conftest import fake_embedding

DOC1 = "DOC1: Alpha 是独占实体, Shared 是共享实体。"
DOC2 = "DOC2: Beta 是独占实体, Shared 是共享实体。"

# 每个文档 1 个 chunk (内容短, 远小于 chunk_token_size)
CHUNK1 = compute_mdhash_id(f"{compute_mdhash_id(DOC1, prefix='doc-')}-0", prefix="chunk-")
CHUNK2 = compute_mdhash_id(f"{compute_mdhash_id(DOC2, prefix='doc-')}-0", prefix="chunk-")


def make_llm():
    """按文档内容返回不同抽取结果的 fake LLM。"""

    async def llm(prompt, system_prompt=None, history_messages=None, stream=False):
        sys = system_prompt or ""
        if "Knowledge Graph Specialist" in sys:
            if "missed or incorrectly formatted" in prompt:
                return "<|COMPLETE|>"  # gleaning 无补抽
            if "DOC1" in prompt:
                return (
                    "entity<|#|>Alpha<|#|>Concept<|#|>仅文档一的实体\n"
                    "entity<|#|>Shared<|#|>Concept<|#|>共享实体\n"
                    "relation<|#|>Alpha<|#|>Shared<|#|>link<|#|>文档一独占关系\n"
                    "relation<|#|>Shared<|#|>Beta<|#|>link<|#|>共享关系\n"
                    "<|COMPLETE|>"
                )
            if "DOC2" in prompt:
                return (
                    "entity<|#|>Beta<|#|>Concept<|#|>仅文档二的实体\n"
                    "entity<|#|>Shared<|#|>Concept<|#|>共享实体\n"
                    "relation<|#|>Shared<|#|>Beta<|#|>link<|#|>共享关系\n"
                    "<|COMPLETE|>"
                )
            return "<|COMPLETE|>"
        if "high_level_keywords" in prompt:
            return '{"high_level_keywords": [], "low_level_keywords": []}'
        if "synthesize" in prompt:
            return "摘要"
        return "回答"

    return llm


@pytest.fixture()
async def rag(tmp_path):
    config = FusionRAGConfig(
        working_dir=str(tmp_path / "ws"),
        llm_func=make_llm(),
        embedding_func=fake_embedding,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        force_llm_summary_on_merge=99,
        cosine_threshold=0.0,
    )
    rag = FusionRAG(config)
    await rag.ainsert(DOC1)
    await rag.ainsert(DOC2)
    return rag


async def test_insert_precondition(rag):
    """前置验证: Alpha/Beta 各占 1 源, Shared/Beta边 共享 2 源。"""
    assert (await rag.graph.get_node("Alpha"))["source_id"] == CHUNK1
    assert (await rag.graph.get_node("Shared"))["source_id"] == f"{CHUNK1}{GRAPH_FIELD_SEP}{CHUNK2}"
    edge = await rag.graph.get_edge("Beta", "Shared")
    assert edge["weight"] == 2.0
    assert edge["source_id"] == f"{CHUNK1}{GRAPH_FIELD_SEP}{CHUNK2}"


async def test_delete_doc_keeps_shared_consistent(rag):
    stats = await rag.adelete_by_doc_id(compute_mdhash_id(DOC1, prefix="doc-"))
    assert stats["chunks_deleted"] == 1
    assert stats["entities_deleted"] == 1   # Alpha
    assert stats["entities_updated"] == 2   # Shared, Beta
    assert stats["relations_deleted"] == 1  # Alpha-Shared (残留边连带)
    assert stats["relations_updated"] == 1  # Shared-Beta

    # 独占实体真删: 图节点 / 实体向量 / 账本 全部消失
    assert await rag.graph.get_node("Alpha") is None
    assert await rag.entities_vdb.get_by_id(compute_mdhash_id("Alpha", prefix="ent-")) is None
    assert await rag.entity_chunks.get_by_id("Alpha") is None

    # 共享实体保留, source_id 扣减
    shared = await rag.graph.get_node("Shared")
    assert shared is not None and shared["source_id"] == CHUNK2
    assert (await rag.entity_chunks.get_by_id("Shared"))["chunk_ids"] == [CHUNK2]

    # 独占关系真删: 图边 / 关系向量(正反 id) / 账本 全部消失
    assert await rag.graph.get_edge("Alpha", "Shared") is None
    rel_id = compute_mdhash_id("AlphaShared", prefix="rel-")
    assert await rag.relationships_vdb.get_by_id(rel_id) is None
    assert await rag.relation_chunks.get_by_id(GRAPH_FIELD_SEP.join(["Alpha", "Shared"])) is None

    # 共享关系保留, weight 重算为剩余源计数
    edge = await rag.graph.get_edge("Beta", "Shared")
    assert edge["weight"] == 1.0
    assert edge["source_id"] == CHUNK2
    assert (await rag.relation_chunks.get_by_id(GRAPH_FIELD_SEP.join(["Beta", "Shared"])))["chunk_ids"] == [CHUNK2]

    # chunk 与文档记录删除
    assert await rag.text_chunks.get_by_id(CHUNK1) is None
    assert await rag.chunks_vdb.get_by_id(CHUNK1) is None
    assert await rag.full_docs.get_by_id(compute_mdhash_id(DOC1, prefix="doc-")) is None
    # 文档二的 chunk 不受影响
    assert await rag.text_chunks.get_by_id(CHUNK2) is not None


async def test_delete_is_idempotent(rag):
    doc1_id = compute_mdhash_id(DOC1, prefix="doc-")
    await rag.adelete_by_doc_id(doc1_id)
    # 重复删除同一 doc: 文档不存在, 抛 KeyError
    with pytest.raises(KeyError):
        await rag.adelete_by_doc_id(doc1_id)
    # 共享实体仍然一致
    assert (await rag.graph.get_node("Shared"))["source_id"] == CHUNK2


async def test_delete_then_reinsert_consistent(rag):
    doc1_id = compute_mdhash_id(DOC1, prefix="doc-")
    await rag.adelete_by_doc_id(doc1_id)
    # 重新导入同文档 (LLM 缓存命中, 抽取结果一致)
    result = await rag.ainsert(DOC1)
    assert result["status"] == "PROCESSED"
    # 数据恢复到删除前状态 (source_id 集合一致, 顺序按 KEEP 旧值优先策略可能不同)
    assert (await rag.graph.get_node("Alpha"))["source_id"] == CHUNK1
    shared_ids = set((await rag.graph.get_node("Shared"))["source_id"].split(GRAPH_FIELD_SEP))
    assert shared_ids == {CHUNK1, CHUNK2}
    edge = await rag.graph.get_edge("Beta", "Shared")
    assert edge["weight"] == 2.0


async def test_delete_nonexistent_doc(rag):
    with pytest.raises(KeyError):
        await rag.adelete_by_doc_id("doc-does-not-exist")

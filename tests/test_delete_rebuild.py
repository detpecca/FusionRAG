"""删除文档时的描述重建测试: 存活实体/关系的描述与向量用剩余 chunk 重合并,
被删文档的语义信息随之清除; 开关关闭时回退为仅重写 source_id/weight。"""

import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG
from fusionrag.prompts import GRAPH_FIELD_SEP
from fusionrag.utils import compute_mdhash_id

from .conftest import fake_embedding

DOC1 = "DOC1: Alpha 是独占实体, Shared 是共享实体。"
DOC2 = "DOC2: Beta 是独占实体, Shared 是共享实体。"

CHUNK1 = compute_mdhash_id(f"{compute_mdhash_id(DOC1, prefix='doc-')}-0", prefix="chunk-")
CHUNK2 = compute_mdhash_id(f"{compute_mdhash_id(DOC2, prefix='doc-')}-0", prefix="chunk-")


def make_llm():
    """两篇文档对共享实体/关系给出不同描述, 便于验证重建后只剩存活文档的语义。"""

    async def llm(prompt, system_prompt=None, history_messages=None, stream=False):
        sys = system_prompt or ""
        if "Knowledge Graph Specialist" in sys:
            if "missed or incorrectly formatted" in prompt:
                return "<|COMPLETE|>"  # gleaning 无补抽
            if "DOC1" in prompt:
                return (
                    "entity<|#|>Alpha<|#|>Concept<|#|>仅文档一的实体\n"
                    "entity<|#|>Shared<|#|>Concept<|#|>文档一视角的共享实体\n"
                    "relation<|#|>Shared<|#|>Beta<|#|>link1<|#|>文档一视角的共享关系\n"
                    "<|COMPLETE|>"
                )
            if "DOC2" in prompt:
                return (
                    "entity<|#|>Beta<|#|>Concept<|#|>仅文档二的实体\n"
                    "entity<|#|>Shared<|#|>Concept<|#|>文档二视角的共享实体\n"
                    "relation<|#|>Shared<|#|>Beta<|#|>link2<|#|>文档二视角的共享关系\n"
                    "<|COMPLETE|>"
                )
            return "<|COMPLETE|>"
        if "high_level_keywords" in prompt:
            return '{"high_level_keywords": [], "low_level_keywords": []}'
        if "synthesize" in prompt:
            return "摘要"
        return "回答"

    return llm


def make_config(tmp_path, rebuild: bool):
    return FusionRAGConfig(
        working_dir=str(tmp_path / "ws"),
        llm_func=make_llm(),
        embedding_func=fake_embedding,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        force_llm_summary_on_merge=99,
        cosine_threshold=0.0,
        delete_rebuild_descriptions=rebuild,
    )


@pytest.fixture()
async def rag(tmp_path):
    rag = FusionRAG(make_config(tmp_path, rebuild=True))
    await rag.ainsert(DOC1)
    await rag.ainsert(DOC2)
    return rag


async def test_delete_rebuilds_shared_descriptions(rag):
    # 删除前: 共享实体描述含两篇文档的碎片, 关系 keywords 为合集
    shared_before = await rag.graph.get_node("Shared")
    assert "文档一视角" in shared_before["description"]
    assert "文档二视角" in shared_before["description"]
    edge_before = await rag.graph.get_edge("Beta", "Shared")
    assert edge_before["keywords"] == "link1,link2"

    stats = await rag.adelete_by_doc_id(compute_mdhash_id(DOC1, prefix="doc-"))
    assert stats["entities_updated"] == 2      # Shared, Beta
    assert stats["entities_rebuilt"] == 2
    assert stats["relations_updated"] == 1     # Shared-Beta
    assert stats["relations_rebuilt"] == 1

    # 共享实体: 描述只剩文档二语义, 文档一碎片被清除
    shared = await rag.graph.get_node("Shared")
    assert shared["description"] == "文档二视角的共享实体"
    assert shared["source_id"] == CHUNK2

    # 共享关系: 描述与 keywords 按剩余 chunk 重建, weight 重算
    edge = await rag.graph.get_edge("Beta", "Shared")
    assert edge["description"] == "文档二视角的共享关系"
    assert edge["keywords"] == "link2"
    assert edge["weight"] == 1.0

    # 向量同步重建: embedding 文本 = name + "\n" + description
    ent_vec = await rag.entities_vdb.get_by_id(compute_mdhash_id("Shared", prefix="ent-"))
    assert ent_vec["content"] == "Shared\n文档二视角的共享实体"
    rel_vec = await rag.relationships_vdb.get_by_id(
        compute_mdhash_id("BetaShared", prefix="rel-")
    )
    assert "文档二视角的共享关系" in rel_vec["content"]
    assert "文档一视角" not in rel_vec["content"]


async def test_delete_then_reinsert_restores(rag):
    """重建路径不破坏幂等性: 删后重导恢复双源状态。"""
    doc1_id = compute_mdhash_id(DOC1, prefix="doc-")
    await rag.adelete_by_doc_id(doc1_id)
    await rag.ainsert(DOC1)
    shared = await rag.graph.get_node("Shared")
    assert set(shared["source_id"].split(GRAPH_FIELD_SEP)) == {CHUNK1, CHUNK2}
    assert "文档一视角" in shared["description"]
    assert "文档二视角" in shared["description"]


async def test_rebuild_disabled_falls_back_to_structural(tmp_path):
    """开关关闭: 保留旧行为 (仅重写 source_id/weight, 描述不动)。"""
    rag = FusionRAG(make_config(tmp_path, rebuild=False))
    await rag.ainsert(DOC1)
    await rag.ainsert(DOC2)

    stats = await rag.adelete_by_doc_id(compute_mdhash_id(DOC1, prefix="doc-"))
    assert stats["entities_rebuilt"] == 0
    assert stats["relations_rebuilt"] == 0

    shared = await rag.graph.get_node("Shared")
    assert shared["source_id"] == CHUNK2            # provenance 仍然干净
    assert "文档一视角" in shared["description"]    # 描述残留 (structural fallback)

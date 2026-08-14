"""合并/去重测试: 描述 <SEP> 拼接去重、source_id 归并、weight 防重复累加、无向边归一。"""

from fusionrag.merge import merge_nodes_and_edges
from fusionrag.prompts import GRAPH_FIELD_SEP
from fusionrag.utils import compute_mdhash_id


def _entity(name, desc, src):
    return {"entity_name": name, "entity_type": "concept", "description": desc, "source_id": src}


def _relation(src, tgt, desc, kw, chunk):
    return {"src_id": src, "tgt_id": tgt, "weight": 1.0, "description": desc, "keywords": kw, "source_id": chunk}


async def _merge(rag, nodes, edges):
    return await merge_nodes_and_edges(
        nodes, edges, rag.graph, rag.entities_vdb, rag.relationships_vdb,
        rag.entity_chunks, rag.relation_chunks, rag.llm, rag.config,
    )


async def test_entity_merge_dedup(rag):
    await _merge(rag, {"Alpha": [_entity("Alpha", "描述一", "chunk-1")]}, {})
    node = await rag.graph.get_node("Alpha")
    assert node["description"] == "描述一"

    # 第二次合并: 重复 chunk-1 (模拟重跑) + 新 chunk-2
    await _merge(
        rag,
        {"Alpha": [_entity("Alpha", "描述一", "chunk-1"), _entity("Alpha", "描述二", "chunk-2")]},
        {},
    )
    node = await rag.graph.get_node("Alpha")
    # 重复描述被精确去重, 新描述 <SEP> 拼接
    assert node["description"] == f"描述一{GRAPH_FIELD_SEP}描述二"
    assert node["source_id"] == f"chunk-1{GRAPH_FIELD_SEP}chunk-2"

    # 实体向量已写入 (embedding 文本 = name\ndescription)
    ent = await rag.entities_vdb.get_by_id(compute_mdhash_id("Alpha", prefix="ent-"))
    assert ent is not None and ent["content"].startswith("Alpha\n")

    # chunk 账本完整保留
    ledger = await rag.entity_chunks.get_by_id("Alpha")
    assert ledger["chunk_ids"] == ["chunk-1", "chunk-2"]


async def test_relation_merge_weight_and_undirected(rag):
    records = [_relation("A", "B", "d1", "k2,k1", "chunk-1")]
    await _merge(rag, {}, {("A", "B"): records})
    edge = await rag.graph.get_edge("A", "B")
    assert edge["weight"] == 1.0
    assert edge["keywords"] == "k1,k2"          # 字典序合并
    assert await rag.graph.has_node("A")        # 缺失端点自动补占位节点

    # 重跑同 chunk: weight 不重复累加
    await _merge(rag, {}, {("A", "B"): records})
    edge = await rag.graph.get_edge("A", "B")
    assert edge["weight"] == 1.0

    # 反向写 (B,A) 无向归一到同一条边, 新 chunk 使 weight 累加
    await _merge(rag, {}, {("B", "A"): [_relation("B", "A", "d2", "k3", "chunk-2")]})
    edge = await rag.graph.get_edge("A", "B")
    assert edge["weight"] == 2.0
    assert edge["source_id"] == f"chunk-1{GRAPH_FIELD_SEP}chunk-2"
    assert edge["keywords"] == "k1,k2,k3"
    assert edge["description"] == f"d1{GRAPH_FIELD_SEP}d2"

    # 关系向量已写入
    rel = await rag.relationships_vdb.get_by_id(compute_mdhash_id("AB", prefix="rel-"))
    assert rel is not None and rel["meta"] == {"src_id": "A", "tgt_id": "B"}


async def test_llm_summary_triggered_when_too_many_parts(rag):
    # force_llm_summary_on_merge=1: 片段数 >= 1 即触发摘要
    rag.config.force_llm_summary_on_merge = 1
    await _merge(rag, {"Beta": [_entity("Beta", "片段一", "chunk-1"), _entity("Beta", "片段二", "chunk-2")]}, {})
    node = await rag.graph.get_node("Beta")
    assert node["description"] == "合并后的摘要描述。"  # FakeLLM 的摘要输出

"""图存储批量方法测试: node_degrees_batch / get_nodes_edges_batch。"""

from __future__ import annotations

from fusionrag.storage import NetworkXGraphStorage


async def _build(tmp_path) -> NetworkXGraphStorage:
    g = NetworkXGraphStorage("g", str(tmp_path))
    await g.upsert_node("A", {"entity_type": "x"})
    await g.upsert_node("B", {"entity_type": "y"})
    await g.upsert_node("C", {"entity_type": "z"})
    await g.upsert_edge("A", "B", {"weight": 1.0})
    await g.upsert_edge("A", "C", {"weight": 2.0})
    return g


async def test_node_degrees_batch(tmp_path):
    g = await _build(tmp_path)
    degrees = await g.node_degrees_batch(["A", "B", "C", "MISSING"])
    assert degrees["A"] == 2
    assert degrees["B"] == 1
    assert degrees["C"] == 1
    assert degrees["MISSING"] == 0  # 不存在的节点度数为 0, 不报错


async def test_node_degrees_batch_matches_single(tmp_path):
    """批量结果应与逐个 node_degree 一致。"""
    g = await _build(tmp_path)
    names = ["A", "B", "C"]
    batch = await g.node_degrees_batch(names)
    single = {n: await g.node_degree(n) for n in names}
    assert batch == single


async def test_get_nodes_edges_batch(tmp_path):
    g = await _build(tmp_path)
    edges = await g.get_nodes_edges_batch(["A", "B", "MISSING"])
    # A 有两条边 (A,B) (A,C); 端点顺序由 networkx 决定, 用集合比较
    assert {frozenset(p) for p in edges["A"]} == {frozenset(("A", "B")), frozenset(("A", "C"))}
    assert {frozenset(p) for p in edges["B"]} == {frozenset(("A", "B"))}
    assert edges["MISSING"] == []


async def test_get_nodes_edges_batch_matches_single(tmp_path):
    g = await _build(tmp_path)
    names = ["A", "B", "C"]
    batch = await g.get_nodes_edges_batch(names)
    for n in names:
        single = await g.get_node_edges(n)
        assert {frozenset(p) for p in batch[n]} == {frozenset(p) for p in single}


async def test_empty_input(tmp_path):
    g = await _build(tmp_path)
    assert await g.node_degrees_batch([]) == {}
    assert await g.get_nodes_edges_batch([]) == {}

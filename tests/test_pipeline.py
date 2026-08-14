"""端到端链路测试: 文档导入(切分/抽取/合并/入库) -> 各模式查询 -> 流式输出。"""

from fusionrag.query import QueryParam

from .conftest import SAMPLE_DOC


async def test_insert_pipeline(rag):
    result = await rag.ainsert(SAMPLE_DOC)
    assert result["status"] == "PROCESSED"
    assert result["chunks"] >= 1
    # 3 个首轮实体 + 1 个 gleaning 补抽实体
    assert result["entities"] == 4
    assert result["relations"] == 2

    # 图谱内容验证
    node = await rag.graph.get_node("星尘科技")
    assert node is not None and node["entity_type"] == "organization"
    edge = await rag.graph.get_edge("云脑平台", "星尘科技")
    assert edge is not None and edge["weight"] >= 1.0

    # 幂等: 相同内容重复导入直接跳过
    again = await rag.ainsert(SAMPLE_DOC)
    assert again["status"] == "SKIPPED"


async def test_naive_query(rag):
    await rag.ainsert(SAMPLE_DOC)
    res = await rag.aquery("星尘科技是什么?", QueryParam(mode="naive"))
    assert res.content == "这是基于知识库的回答。"
    assert res.raw_data["chunks"], "naive 模式应检索到文本片段"


async def test_hybrid_query(rag):
    await rag.ainsert(SAMPLE_DOC)
    res = await rag.aquery("星尘科技和云脑平台是什么关系?", QueryParam(mode="hybrid"))
    assert res.content == "这是基于知识库的回答。"
    data = res.raw_data
    assert data["keywords"]["high_level"] == ["图增强检索", "知识图谱"]
    assert data["keywords"]["low_level"] == ["星尘科技", "云脑平台"]
    assert "星尘科技" in data["entities"]
    assert data["relations"], "hybrid 模式应检索到关系"
    assert data["chunks"], "hybrid 模式应检索到文本片段"


async def test_local_and_global_query(rag):
    await rag.ainsert(SAMPLE_DOC)
    local = await rag.aquery("星尘科技", QueryParam(mode="local"))
    assert "星尘科技" in local.raw_data["entities"]
    global_res = await rag.aquery("图增强检索 知识图谱", QueryParam(mode="global"))
    assert global_res.raw_data["relations"]


async def test_stream_query(rag):
    await rag.ainsert(SAMPLE_DOC)
    res = await rag.aquery("介绍云脑平台", QueryParam(mode="hybrid", stream=True))
    assert res.is_streaming
    parts = [p async for p in res.response_iterator]
    assert "".join(parts) == "这是基于知识库的回答。"


async def test_only_need_context(rag):
    await rag.ainsert(SAMPLE_DOC)
    res = await rag.aquery("星尘科技", QueryParam(mode="hybrid", only_need_context=True))
    assert "星尘科技" in res.context
    assert "Entities" in res.context

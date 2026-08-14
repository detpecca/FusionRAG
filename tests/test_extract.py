"""抽取解析测试: 分隔符格式解析、字段规范化、gleaning 合并。"""

from fusionrag.extract import (
    _handle_entity,
    _handle_relation,
    _merge_gleaning,
    _process_extraction_result,
    extract_entities,
)
from fusionrag.llm import LLMService


def test_parse_records():
    text = (
        "entity<|#|>云脑平台<|#|>Artifact<|#|>一个图增强检索生成系统。\n"
        "relation<|#|>云脑平台<|#|>知识图谱<|#|>usage, storage<|#|>使用知识图谱存储实体与关系。\n"
        "<|COMPLETE|>\nentity<|#|>不应出现<|#|>Other<|#|>在完成标记之后。"
    )
    nodes, edges = _process_extraction_result(text, "chunk-1")
    assert list(nodes) == ["云脑平台"]
    assert nodes["云脑平台"][0]["entity_type"] == "artifact"  # type 小写规范化
    assert nodes["云脑平台"][0]["source_id"] == "chunk-1"
    assert ("云脑平台", "知识图谱") in edges
    edge = edges[("云脑平台", "知识图谱")][0]
    assert edge["weight"] == 1.0
    assert edge["keywords"] == "usage, storage"


def test_parse_garbage_lines_skipped():
    text = "这不是抽取记录\nentity<|#|>只有名字\n\nrelation<|#|>A<|#|>B<|#|>kw<|#|>desc"
    nodes, edges = _process_extraction_result(text, "chunk-1")
    assert nodes == {}                       # 字段数不够的 entity 行被丢弃
    assert list(edges) == [("A", "B")]


def test_handle_entity_validation():
    assert _handle_entity("", "Concept", "d", "c") is None        # 空名
    assert _handle_entity("12345", "Concept", "d", "c") is None   # 纯数字
    assert _handle_entity("x" * 300, "Concept", "d", "c") is None  # 超长
    rec = _handle_entity('"阿里, 巴巴"', "Organization,Company", "d", "c")
    assert rec is not None
    assert rec["entity_type"] == "organization"  # 逗号取首个


def test_handle_relation_validation():
    assert _handle_relation("A", "A", "kw", "d", "c") is None     # 自环丢弃
    assert _handle_relation("", "B", "kw", "d", "c") is None
    rec = _handle_relation("A", "B", "关键词，测试", "d", "c")
    assert rec["keywords"] == "关键词,测试"                      # 全角逗号转半角


def test_gleaning_merge_keeps_longer_description():
    nodes = {"E": [{"entity_name": "E", "entity_type": "concept", "description": "短", "source_id": "c1"}]}
    edges = {("A", "B"): [{"src_id": "A", "tgt_id": "B", "weight": 1.0, "description": "短", "keywords": "k", "source_id": "c1"}]}
    g_nodes = {
        "E": [{"entity_name": "E", "entity_type": "concept", "description": "更长的描述内容", "source_id": "c1"}],
        "N": [{"entity_name": "N", "entity_type": "concept", "description": "新实体", "source_id": "c1"}],
    }
    g_edges = {("A", "B"): [{"src_id": "A", "tgt_id": "B", "weight": 1.0, "description": "更长的关系描述", "keywords": "k", "source_id": "c1"}]}
    nodes, edges = _merge_gleaning(nodes, edges, g_nodes, g_edges)
    assert nodes["E"][0]["description"] == "更长的描述内容"   # 长描述覆盖
    assert "N" in nodes                                     # 新实体并入
    assert edges[("A", "B")][0]["description"] == "更长的关系描述"


async def test_extract_entities_with_gleaning(fake_llm, test_config):
    llm = LLMService(test_config)
    chunks = {"chunk-a": {"content": "第一段"}, "chunk-b": {"content": "第二段"}}
    nodes, edges = await extract_entities(chunks, llm, test_config)
    # 首轮实体在两个 chunk 都出现 -> 聚合为 2 条记录
    assert len(nodes["星尘科技"]) == 2
    # gleaning 补抽的新实体也进来了
    assert "向量检索" in nodes
    assert nodes["向量检索"][0]["source_id"] in ("chunk-a", "chunk-b")
    # 关系按 (src, tgt) 聚合
    assert ("星尘科技", "云脑平台") in edges
    assert len(edges[("星尘科技", "云脑平台")]) == 2

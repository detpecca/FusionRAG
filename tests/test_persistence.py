"""持久化测试: 存储落盘后新实例能完整加载 (文档幂等/图谱/向量/会话)。"""

from fusionrag.core import FusionRAG
from fusionrag.query import QueryParam

from .conftest import SAMPLE_DOC


async def test_persistence_across_instances(test_config):
    rag1 = FusionRAG(test_config)
    result = await rag1.ainsert(SAMPLE_DOC)
    assert result["status"] == "PROCESSED"
    await rag1.sessions.append_turn("s1", "问题一", "回答一")

    # 同 working_dir 新建实例, 模拟服务重启
    rag2 = FusionRAG(test_config)

    # 图谱节点与边还在
    node = await rag2.graph.get_node("星尘科技")
    assert node is not None and node["entity_type"] == "organization"
    edge = await rag2.graph.get_edge("云脑平台", "星尘科技")
    assert edge is not None

    # 向量库还在, 可以直接检索
    res = await rag2.aquery("星尘科技是什么?", QueryParam(mode="naive"))
    assert res.raw_data["chunks"]

    # 文档状态还在, 重复导入幂等跳过
    again = await rag2.ainsert(SAMPLE_DOC)
    assert again["status"] == "SKIPPED"

    # 会话历史还在
    history = await rag2.sessions.get_history("s1")
    assert history == [
        {"role": "user", "content": "问题一"},
        {"role": "assistant", "content": "回答一"},
    ]

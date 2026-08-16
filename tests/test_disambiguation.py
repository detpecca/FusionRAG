"""实体消歧测试: 别名归一 / 别名表持久化复用 / 自环边清理 / 删除级联清理。"""

from __future__ import annotations

import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG
from fusionrag.disambiguate import resolve_entity_aliases
from fusionrag.llm import LLMService
from fusionrag.storage import JsonKVStorage

from .conftest import CANNED_EXTRACTION, SAMPLE_DOC, FakeLLM, fake_embedding


def _rec(desc: str = "描述") -> dict:
    return {
        "entity_type": "Organization",
        "description": desc,
        "source_id": "chunk-x",
        "weight": 1.0,
    }


def _cfg(fake_llm, **kw) -> FusionRAGConfig:
    return FusionRAGConfig(
        llm_func=fake_llm,
        embedding_func=fake_embedding,
        entity_disambiguation=True,
        **kw,
    )


async def test_alias_merged_and_persisted(tmp_path):
    """别名经 LLM 确认后归一到规范名, 映射持久化; 二次导入直接命中别名表。"""
    llm = FakeLLM()
    llm.alias_pairs = [{"星尘科技", "星尘科技集团"}]
    kv = JsonKVStorage("entity_aliases", str(tmp_path))
    nodes = {"星尘科技集团": [_rec()]}
    edges = {("星尘科技集团", "云脑平台"): [_rec("关系")]}

    n2, e2, m = await resolve_entity_aliases(
        nodes, edges, ["星尘科技", "云脑平台"], kv,
        LLMService(_cfg(llm)), _cfg(llm),
    )
    assert m == {"星尘科技集团": "星尘科技"}
    assert set(n2) == {"星尘科技"}
    assert set(e2) == {("星尘科技", "云脑平台")}
    assert await kv.get_by_id("星尘科技集团") == {"canonical": "星尘科技"}
    await kv.index_done_callback()  # 真实链路由 core._flush 落盘, 单测手动模拟

    # 二次导入: 别名表精确命中, 不再触发任何 LLM 别名确认
    llm2 = FakeLLM()
    kv2 = JsonKVStorage("entity_aliases", str(tmp_path))  # 模拟重启
    n3, e3, m3 = await resolve_entity_aliases(
        {"星尘科技集团": [_rec()]}, {}, ["星尘科技"], kv2,
        LLMService(_cfg(llm2)), _cfg(llm2),
    )
    assert m3 == {"星尘科技集团": "星尘科技"}
    assert set(n3) == {"星尘科技"}
    assert not any("Alias Judge" in (c.get("system_prompt") or "") for c in llm2.calls)


async def test_no_merge_when_llm_says_no(tmp_path):
    """相似度达阈值但 LLM 判非同一实体: 保持分离, 不写别名表。"""
    llm = FakeLLM()  # alias_pairs 为空 -> 一律 NO
    kv = JsonKVStorage("entity_aliases", str(tmp_path))
    nodes = {"星尘科技集团": [_rec()]}

    n2, e2, m = await resolve_entity_aliases(
        nodes, {}, ["星尘科技"], kv, LLMService(_cfg(llm)), _cfg(llm)
    )
    assert m == {}
    assert set(n2) == {"星尘科技集团"}
    assert await kv.get_by_id("星尘科技集团") is None


async def test_self_loop_edge_dropped(tmp_path):
    """消歧后两端同一实体的边成为自环, 直接丢弃。"""
    llm = FakeLLM()
    llm.alias_pairs = [{"华东科技", "华东科技公司"}]
    kv = JsonKVStorage("entity_aliases", str(tmp_path))
    nodes = {"华东科技公司": [_rec()], "星河": [_rec()]}
    edges = {("华东科技公司", "华东科技"): [_rec("自环")], ("华东科技公司", "星河"): [_rec("正常")]}

    n2, e2, m = await resolve_entity_aliases(
        nodes, edges, ["华东科技"], kv, LLMService(_cfg(llm)), _cfg(llm)
    )
    assert m == {"华东科技公司": "华东科技"}
    assert set(e2) == {("华东科技", "星河")}


async def test_new_new_alias_in_same_batch(tmp_path):
    """同批两个新实体互为别名: 归一到字典序较小的规范名。"""
    llm = FakeLLM()
    llm.alias_pairs = [{"云脑平台", "云脑平台系统"}]
    kv = JsonKVStorage("entity_aliases", str(tmp_path))
    nodes = {"云脑平台": [_rec("a")], "云脑平台系统": [_rec("b")]}

    n2, e2, m = await resolve_entity_aliases(
        nodes, {}, [], kv, LLMService(_cfg(llm)), _cfg(llm)
    )
    assert m == {"云脑平台系统": "云脑平台"}
    assert set(n2) == {"云脑平台"}
    assert len(n2["云脑平台"]) == 2  # 两条记录并入规范名


async def test_disabled_by_config(tmp_path):
    llm = FakeLLM()
    kv = JsonKVStorage("entity_aliases", str(tmp_path))
    cfg = FusionRAGConfig(
        llm_func=llm, embedding_func=fake_embedding, entity_disambiguation=False
    )
    nodes = {"星尘科技集团": [_rec()]}
    n2, e2, m = await resolve_entity_aliases(
        nodes, {}, ["星尘科技"], kv, LLMService(cfg), cfg
    )
    assert m == {}
    assert set(n2) == {"星尘科技集团"}


# ---------------------------------------------------------------------------
# 集成: 完整导入链路中的消歧
# ---------------------------------------------------------------------------

ALT_DOC = SAMPLE_DOC + "星尘科技集团的研发团队持续迭代云脑平台。"

ALT_EXTRACTION = (
    CANNED_EXTRACTION.replace("<|COMPLETE|>", "")
    + "entity<|#|>星尘科技集团<|#|>Organization<|#|>星尘科技集团的别名描述。\n"
    + "relation<|#|>星尘科技集团<|#|>云脑平台<|#|>develop<|#|>星尘科技集团研发云脑平台。\n"
    + "<|COMPLETE|>"
)


class AliasFakeLLM(FakeLLM):
    """第二个文档 (含 '集团' 标记) 的抽取产出别名实体 星尘科技集团。"""

    def __init__(self) -> None:
        super().__init__()
        self.alias_pairs = [{"星尘科技", "星尘科技集团"}]

    async def __call__(self, prompt, system_prompt=None, history_messages=None, stream=False):
        sys_ = system_prompt or ""
        if "Knowledge Graph Specialist" in sys_ and "集团" in prompt:
            if "missed or incorrectly formatted" in prompt:
                return "<|COMPLETE|>"  # 补抽无新增
            return ALT_EXTRACTION
        return await super().__call__(prompt, system_prompt, history_messages, stream)


@pytest.fixture()
def alias_config(tmp_path):
    return FusionRAGConfig(
        working_dir=str(tmp_path / "workspace"),
        llm_func=AliasFakeLLM(),
        embedding_func=fake_embedding,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        summary_max_tokens=2000,
        force_llm_summary_on_merge=99,
        cosine_threshold=0.0,
        entity_disambiguation=True,
    )


async def test_insert_merge_alias_then_delete_cascade(alias_config):
    rag = FusionRAG(alias_config)
    await rag.ainsert(SAMPLE_DOC)
    r2 = await rag.ainsert(ALT_DOC)
    assert r2["status"] == "PROCESSED"

    # 别名实体被归一: 图中只有规范名
    assert await rag.graph.has_node("星尘科技")
    assert not await rag.graph.has_node("星尘科技集团")
    assert await rag.entity_aliases.get_by_id("星尘科技集团") == {"canonical": "星尘科技"}
    # 别名记录的 chunk 登进了规范名账本
    ledger = await rag.entity_chunks.get_by_id("星尘科技")
    assert len(ledger["chunk_ids"]) == 2

    # 删除别名所在文档: 规范实体存活 (另一文档仍有贡献), 别名映射保留
    doc2_id = r2["doc_id"]
    await rag.adelete_by_doc_id(doc2_id)
    assert await rag.graph.has_node("星尘科技")
    assert await rag.entity_aliases.get_by_id("星尘科技集团") == {"canonical": "星尘科技"}

    # 删除最后一个文档: 实体真删, 别名映射随之清理
    docs = await rag.alist_documents()
    last_id = docs[0]["doc_id"]
    await rag.adelete_by_doc_id(last_id)
    assert not await rag.graph.has_node("星尘科技")
    assert await rag.entity_aliases.get_by_id("星尘科技集团") is None

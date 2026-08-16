"""测试基础设施: 确定性 Fake LLM / Fake Embedding, 不依赖外部 API。"""

from __future__ import annotations

import re

import numpy as np
import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG

# 固定的抽取输出: 无论输入什么都抽取这组实体/关系, 便于断言
CANNED_EXTRACTION = (
    "entity<|#|>星尘科技<|#|>Organization<|#|>星尘科技是一家人工智能公司。\n"
    "entity<|#|>云脑平台<|#|>Artifact<|#|>云脑平台是一个图增强检索生成系统。\n"
    "entity<|#|>知识图谱<|#|>Concept<|#|>知识图谱用于组织实体与关系。\n"
    "relation<|#|>星尘科技<|#|>云脑平台<|#|>develop, product<|#|>星尘科技自主研发了云脑平台。\n"
    "relation<|#|>云脑平台<|#|>知识图谱<|#|>usage, storage<|#|>云脑平台使用知识图谱存储实体与关系。\n"
    "<|COMPLETE|>"
)

GLEANING_EXTRACTION = (
    "entity<|#|>向量检索<|#|>Method<|#|>向量检索是补抽阶段发现的遗漏实体。\n"
    "<|COMPLETE|>"
)


class FakeLLM:
    """按 prompt 特征分发: 抽取 / 关键词 / 摘要 / 问答 / 别名确认。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        # 实体消歧: 出现在同一 prompt 中即回答 YES 的名字对 (测试控制)
        self.alias_pairs: list[set[str]] = []

    async def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        stream: bool = False,
    ):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "stream": stream})
        sys = system_prompt or ""

        # 实体关系抽取 (system prompt 特征)
        if "Knowledge Graph Specialist" in sys:
            if "missed or incorrectly formatted" in prompt:
                return GLEANING_EXTRACTION
            return CANNED_EXTRACTION
        # 实体别名确认 (实体消歧)
        if "Entity Alias Judge" in sys:
            yes = any(
                all(n in prompt for n in pair) for pair in self.alias_pairs
            )
            return "YES" if yes else "NO"
        # 查询关键词提取 (无 system prompt, 内容特征)
        if "high_level_keywords" in prompt:
            return '{"high_level_keywords": ["图增强检索", "知识图谱"], "low_level_keywords": ["星尘科技", "云脑平台"]}'
        # 描述摘要 (无 system prompt, 内容特征)
        if "synthesize" in prompt:
            return "合并后的摘要描述。"
        # 问答 (system prompt 特征)
        if "expert AI assistant" in sys:
            if stream:
                async def gen():
                    for piece in ["这是", "基于知识库", "的回答。"]:
                        yield piece
                return gen()
            return "这是基于知识库的回答。"
        return ""


async def fake_embedding(texts: list[str]) -> np.ndarray:
    """确定性词袋哈希 embedding: 相同词的文本向量相近, 满足检索测试。"""
    dim = 128
    vectors = []
    for text in texts:
        v = np.zeros(dim, dtype=np.float32)
        for token in re.findall(r"[a-zA-Z0-9_]+|[一-鿿]", text.lower()):
            v[hash(token) % dim] += 1.0
        vectors.append(v)
    return np.asarray(vectors, dtype=np.float32)


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def test_config(tmp_path, fake_llm) -> FusionRAGConfig:
    return FusionRAGConfig(
        working_dir=str(tmp_path / "workspace"),
        llm_func=fake_llm,
        embedding_func=fake_embedding,
        chunk_token_size=200,
        chunk_overlap_token_size=20,
        summary_max_tokens=2000,      # 测试默认不触发 LLM 摘要
        force_llm_summary_on_merge=99,
        cosine_threshold=0.0,          # 词袋向量余弦值较低, 测试放宽阈值
    )


@pytest.fixture()
async def rag(test_config) -> FusionRAG:
    return FusionRAG(test_config)


SAMPLE_DOC = (
    "星尘科技是一家人工智能公司, 专注于图增强检索生成技术。"
    "其核心产品云脑平台使用知识图谱组织实体与关系, 并结合向量检索进行增强问答。"
    "知识图谱支持实体抽取、关系抽取与混合检索。"
)

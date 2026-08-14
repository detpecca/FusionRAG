"""embedding 指纹校验: 换模型/维度时拦截, 防静默向量数据损坏。"""

from __future__ import annotations

import json
import os

import pytest

from fusionrag.config import FusionRAGConfig
from fusionrag.core import FusionRAG

from .conftest import fake_embedding


def _config(tmp_path, fake_llm, *, model="text-embedding-3-small", dim=1536):
    return FusionRAGConfig(
        working_dir=str(tmp_path / "ws"),
        llm_func=fake_llm,
        embedding_func=fake_embedding,
        embedding_model=model,
        embedding_dim=dim,
    )


def test_fingerprint_written_on_first_init(tmp_path, fake_llm):
    cfg = _config(tmp_path, fake_llm)
    FusionRAG(cfg)
    path = os.path.join(cfg.working_dir, "embedding_meta.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == {"embedding_model": "text-embedding-3-small", "embedding_dim": 1536}


def test_matching_fingerprint_passes(tmp_path, fake_llm):
    cfg = _config(tmp_path, fake_llm)
    FusionRAG(cfg)
    FusionRAG(cfg)  # 同配置二次初始化应放行, 不抛错


def test_model_mismatch_raises(tmp_path, fake_llm):
    FusionRAG(_config(tmp_path, fake_llm, model="model-a"))
    with pytest.raises(ValueError, match="embedding 配置与存量向量不一致"):
        FusionRAG(_config(tmp_path, fake_llm, model="model-b"))


def test_dim_mismatch_raises(tmp_path, fake_llm):
    FusionRAG(_config(tmp_path, fake_llm, dim=1536))
    with pytest.raises(ValueError, match="embedding 配置与存量向量不一致"):
        FusionRAG(_config(tmp_path, fake_llm, dim=1024))


def test_missing_fingerprint_is_backfilled(tmp_path, fake_llm):
    """旧库无指纹文件时: 按当前配置回填, 不报错(向后兼容)。"""
    cfg = _config(tmp_path, fake_llm)
    os.makedirs(cfg.working_dir, exist_ok=True)
    # 模拟旧库: 目录存在但无 embedding_meta.json
    FusionRAG(cfg)
    assert os.path.exists(os.path.join(cfg.working_dir, "embedding_meta.json"))


def test_corrupt_fingerprint_is_rewritten(tmp_path, fake_llm):
    cfg = _config(tmp_path, fake_llm)
    os.makedirs(cfg.working_dir, exist_ok=True)
    path = os.path.join(cfg.working_dir, "embedding_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ not valid json")
    FusionRAG(cfg)  # 损坏文件应按当前配置重写, 不抛错
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["embedding_model"] == "text-embedding-3-small"

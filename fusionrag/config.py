"""FusionRAG 全局配置, 支持环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

try:
    # 本地启动时自动加载项目根目录的 .env; 不覆盖已存在的环境变量(Docker 的 env_file 优先)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # 未安装 python-dotenv 时降级为仅读系统环境变量
    pass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class FusionRAGConfig:
    """所有可调参数集中在这里, 环境变量可覆盖 (见 from_env / .env.example)。"""

    # ---- 存储 ----
    working_dir: str = "./fusionrag_workspace"
    # 存储后端: 每类存储可独立选后端。storage_backend 为总兜底,
    # 优先级 kv_backend > storage_backend > "json" (vector/graph 同理)。
    storage_backend: str = "json"     # 总兜底: json(默认, 零依赖)
    kv_backend: str = ""              # KV 后端: json / sqlite; 空则回退 storage_backend
    vector_backend: str = ""          # 向量后端: json; 空则回退 storage_backend
    graph_backend: str = ""           # 图后端: json; 空则回退 storage_backend

    def resolve_kv_backend(self) -> str:
        return self.kv_backend or self.storage_backend or "json"

    def resolve_vector_backend(self) -> str:
        return self.vector_backend or self.storage_backend or "json"

    def resolve_graph_backend(self) -> str:
        return self.graph_backend or self.storage_backend or "json"

    # ---- LLM (OpenAI 兼容接口) ----
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_max_async: int = 4            # LLM 并发上限
    llm_timeout: float = 240.0
    llm_temperature: float | None = None   # None = 不传给服务端 (部分模型如 Kimi 仅允许 temperature=1)
    enable_llm_cache: bool = True     # 抽取/摘要结果缓存, 重跑文档时省钱省时
    llm_max_retries: int = 5          # 429/5xx/连接错误的最大重试次数
    retry_base_delay: float = 2.0     # 指数退避基数(秒), 实际等待 = base * 2^attempt + 抖动

    # ---- Embedding (OpenAI 兼容接口) ----
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""       # 为空则复用 llm_api_key
    embedding_base_url: str = ""      # 为空则复用 llm_base_url
    embedding_dim: int = 1536
    embedding_batch_num: int = 10
    embedding_max_async: int = 8

    # ---- Rerank (Jina/Cohere 风格 POST {base_url}/rerank, 可选的精排服务) ----
    rerank_model: str = ""            # 空 = 不启用 rerank (如 gte-rerank-v2 / jina-reranker-v2)
    rerank_api_key: str = ""          # 为空则复用 llm_api_key
    rerank_base_url: str = ""         # 为空则复用 llm_base_url
    rerank_top_n: int = 20            # 精排后保留的 chunk 数
    min_rerank_score: float = 0.0     # 低于该分的 chunk 丢弃
    rerank_timeout: float = 60.0

    # ---- 切分 ----
    chunk_token_size: int = 1200      # 切分窗口 token 数
    chunk_overlap_token_size: int = 100

    # ---- 抽取 ----
    entity_extract_max_gleaning: int = 1   # 补抽轮数
    entity_name_max_length: int = 256
    max_entity_records: int = 40           # 单次抽取实体数量上限
    max_total_records: int = 100           # 单次抽取实体+关系总数上限
    language: str = "Chinese"              # 抽取与回答使用的语言

    # ---- 合并/摘要 ----
    summary_max_tokens: int = 1200         # 描述合并后超过该 token 数则调 LLM 摘要
    force_llm_summary_on_merge: int = 8    # 描述片段数 >= 该值时强制 LLM 摘要
    max_source_ids_per_entity: int = 200

    # ---- 检索 ----
    top_k: int = 40
    chunk_top_k: int = 20
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 30000
    related_chunk_number: int = 5          # 每个实体/关系关联的文本片段数
    cosine_threshold: float = 0.2          # 向量检索余弦相似度下限

    # ---- 会话 ----
    max_history_messages: int = 20         # 长对话滑动窗口: 最多保留的历史消息条数

    # ---- 删除 ----
    delete_rebuild_descriptions: bool = True  # 删除文档时用 LLM 缓存重建存活实体/关系描述

    # 允许 ainsert/aquery 注入自定义函数(测试 mock 用), 不参与 env 加载
    llm_func: object = field(default=None, repr=False)
    embedding_func: object = field(default=None, repr=False)
    rerank_func: object = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "FusionRAGConfig":
        cfg = cls(
            working_dir=_env("WORKING_DIR", cls.working_dir),
            storage_backend=_env("STORAGE_BACKEND", cls.storage_backend),
            kv_backend=_env("KV_BACKEND", cls.kv_backend),
            vector_backend=_env("VECTOR_BACKEND", cls.vector_backend),
            graph_backend=_env("GRAPH_BACKEND", cls.graph_backend),
            llm_model=_env("LLM_MODEL", cls.llm_model),
            llm_api_key=_env("LLM_API_KEY", cls.llm_api_key),
            llm_base_url=_env("LLM_BASE_URL", cls.llm_base_url),
            llm_max_async=_env_int("LLM_MAX_ASYNC", cls.llm_max_async),
            llm_timeout=_env_float("LLM_TIMEOUT", cls.llm_timeout),
            llm_temperature=(
                float(os.environ["LLM_TEMPERATURE"]) if os.environ.get("LLM_TEMPERATURE") else None
            ),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", cls.llm_max_retries),
            retry_base_delay=_env_float("RETRY_BASE_DELAY", cls.retry_base_delay),
            enable_llm_cache=_env("ENABLE_LLM_CACHE", "true").lower() == "true",
            embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
            embedding_api_key=_env("EMBEDDING_API_KEY", ""),
            embedding_base_url=_env("EMBEDDING_BASE_URL", ""),
            embedding_dim=_env_int("EMBEDDING_DIM", cls.embedding_dim),
            rerank_model=_env("RERANK_MODEL", cls.rerank_model),
            rerank_api_key=_env("RERANK_API_KEY", ""),
            rerank_base_url=_env("RERANK_BASE_URL", ""),
            rerank_top_n=_env_int("RERANK_TOP_N", cls.rerank_top_n),
            min_rerank_score=_env_float("MIN_RERANK_SCORE", cls.min_rerank_score),
            rerank_timeout=_env_float("RERANK_TIMEOUT", cls.rerank_timeout),
            chunk_token_size=_env_int("CHUNK_TOKEN_SIZE", cls.chunk_token_size),
            chunk_overlap_token_size=_env_int(
                "CHUNK_OVERLAP_TOKEN_SIZE", cls.chunk_overlap_token_size
            ),
            entity_extract_max_gleaning=_env_int(
                "MAX_GLEANING", cls.entity_extract_max_gleaning
            ),
            language=_env("SUMMARY_LANGUAGE", cls.language),
            top_k=_env_int("TOP_K", cls.top_k),
            chunk_top_k=_env_int("CHUNK_TOP_K", cls.chunk_top_k),
            max_total_tokens=_env_int("MAX_TOTAL_TOKENS", cls.max_total_tokens),
            cosine_threshold=_env_float("COSINE_THRESHOLD", cls.cosine_threshold),
            max_history_messages=_env_int(
                "MAX_HISTORY_MESSAGES", cls.max_history_messages
            ),
            delete_rebuild_descriptions=_env(
                "DELETE_REBUILD_DESCRIPTIONS", "true"
            ).lower() == "true",
        )
        return cfg

    def __post_init__(self) -> None:
        # 校验: 仅检查 dataclass 声明过的字段都被初始化(防止手误)
        for f in fields(self):
            getattr(self, f.name)

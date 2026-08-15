"""LLM / Embedding 客户端封装 (OpenAI 兼容接口), 带 LLM 结果缓存与限流重试。

设计要点:
- llm_func 签名: async (prompt, system_prompt, history_messages, stream) -> str | AsyncIterator
- 抽取/摘要结果按 (model + prompt + system + history) 哈希缓存进 KV 存储
- 并发由 asyncio.Semaphore 限制
- 429 / 5xx / 连接错误按指数退避重试 (应对服务商限流)
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import numpy as np

from .config import FusionRAGConfig
from .utils import compute_args_hash, remove_think_tags

logger = logging.getLogger("fusionrag.llm")


def _is_retryable(exc: Exception) -> bool:
    """429(限流) / 5xx(服务端) / 连接与超时错误 可重试。"""
    status = getattr(exc, "status_code", None)
    if status == 429 or (status is not None and status >= 500):
        return True
    try:
        from openai import APIConnectionError, APITimeoutError

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError))


def _partial_marker_len(text: str, marker: str) -> int:
    """text 后缀中可能是 marker 不完整前缀的最长长度 (用于流式过滤时暂存尾部)。"""
    for k in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:k]):
            return k
    return 0


async def _strip_think_stream(
    chunks: AsyncIterator[str],
) -> AsyncIterator[str]:
    """流式版 remove_think_tags: 过滤推理模型的 <think>...</think>。

    标签可能被拆在多个 delta 里, 用缓冲 + 状态机处理:
    - 未进 think: 暂存可能是 "<think>" 不完整前缀的尾部, 其余立即下发
    - think 内: 只找 "</think>", 之后恢复下发
    """
    OPEN, CLOSE = "<think>", "</think>"
    buf, in_think = "", False
    async for delta in chunks:
        buf += delta
        while True:
            if in_think:
                end = buf.find(CLOSE)
                if end == -1:
                    # 丢弃 think 内容, 但保留可能是 "</think>" 前缀的尾部
                    keep = _partial_marker_len(buf, CLOSE)
                    buf = buf[len(buf) - keep:] if keep else ""
                    break
                buf = buf[end + len(CLOSE):]
                in_think = False
            else:
                start = buf.find(OPEN)
                if start == -1:
                    keep = _partial_marker_len(buf, OPEN)
                    emit = buf[: len(buf) - keep] if keep else buf
                    if emit:
                        yield emit
                    buf = buf[len(buf) - keep:] if keep else ""
                    break
                if start > 0:
                    yield buf[:start]
                buf = buf[start + len(OPEN):]
                in_think = True
    # 流结束: 剩余缓冲 (不含完整标签的残片) 原样下发
    if buf:
        yield buf


class LLMService:
    """统一的 LLM/Embedding 入口。config.llm_func / embedding_func 可注入自定义实现(测试用)。"""

    # 每积累多少条缓存写执行一次落盘
    CACHE_FLUSH_EVERY = 32

    def __init__(self, config: FusionRAGConfig, cache_kv: Any = None) -> None:
        self.config = config
        self._cache_kv = cache_kv
        # 缓存脏计数节流: JSON 后端每次 index_done_callback 全量重写文件,
        # 每次调用都 flush 会造成 O(N^2) 磁盘写放大, 攒够 CACHE_FLUSH_EVERY
        # 条再落盘; 导入结束时 core._flush() 兜底
        self._cache_dirty = 0
        self._llm_sem = asyncio.Semaphore(config.llm_max_async)
        self._emb_sem = asyncio.Semaphore(config.embedding_max_async)
        self._custom_llm = config.llm_func
        self._custom_embed = config.embedding_func
        self._client = None
        self._emb_client = None

        if self._custom_llm is None or self._custom_embed is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "openai package is required for the default LLM/embedding client. "
                    "Install it or inject custom llm_func/embedding_func."
                ) from e
            if self._custom_llm is None:
                self._client = AsyncOpenAI(
                    api_key=config.llm_api_key or "EMPTY",
                    base_url=config.llm_base_url,
                    timeout=config.llm_timeout,
                )
            if self._custom_embed is None:
                self._emb_client = AsyncOpenAI(
                    api_key=config.embedding_api_key or config.llm_api_key or "EMPTY",
                    base_url=config.embedding_base_url or config.llm_base_url,
                    timeout=config.llm_timeout,
                )

    # ------------------------------------------------------------------ retry

    async def _retry(self, func: Callable[[], Awaitable[Any]], what: str) -> Any:
        """指数退避重试: base * 2^attempt + 随机抖动, 仅对可重试错误生效。"""
        max_retries = self.config.llm_max_retries
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                if not _is_retryable(e) or attempt >= max_retries:
                    raise
                delay = self.config.retry_base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "%s 调用失败(%s), %.1fs 后重试 (%d/%d)",
                    what, e, delay, attempt + 1, max_retries,
                )
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------ chat

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list[dict]] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        cache_type: str = "default",
    ) -> str | AsyncIterator[str]:
        """调用 LLM。stream=True 时返回异步迭代器(逐段产出文本), 否则返回完整字符串。"""
        history_messages = history_messages or []
        cache_key: str | None = None
        if (
            use_cache
            and not stream
            and self.config.enable_llm_cache
            and self._cache_kv is not None
        ):
            cache_key = compute_args_hash(
                self.config.llm_model, prompt, system_prompt, history_messages
            )
            cached = await self._cache_kv.get_by_id(cache_key)
            if cached is not None:
                logger.debug("LLM cache hit (%s)", cache_type)
                return cached["content"]

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        async def _do_call() -> Any:
            if self._custom_llm is not None:
                return await self._custom_llm(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    stream=stream,
                )
            return await self._openai_chat(messages, stream, max_tokens)

        result = await self._retry(_do_call, "LLM chat")

        if stream:
            # 流式同样过滤 <think> 标签 (与非流式口径一致)
            return _strip_think_stream(result)  # type: ignore[arg-type]

        content = remove_think_tags(str(result))
        if cache_key is not None and self._cache_kv is not None:
            await self._cache_kv.upsert(
                {cache_key: {"content": content, "cache_type": cache_type}}
            )
            self._cache_dirty += 1
            if self._cache_dirty >= self.CACHE_FLUSH_EVERY:
                await self._cache_kv.index_done_callback()
                self._cache_dirty = 0
        return content

    async def _openai_chat(
        self, messages: list[dict], stream: bool, max_tokens: Optional[int]
    ) -> str | AsyncIterator[str]:
        async with self._llm_sem:
            kwargs: dict[str, Any] = {}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if self.config.llm_temperature is not None:
                kwargs["temperature"] = self.config.llm_temperature
            response = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.config.llm_model,
                messages=messages,
                stream=stream,
                **kwargs,
            )
        if not stream:
            return response.choices[0].message.content or ""

        async def iterator() -> AsyncIterator[str]:
            async for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        return iterator()

    # ------------------------------------------------------------------ embed

    async def embed(self, texts: list[str]) -> np.ndarray:
        """批量 embedding, 返回 (n, dim) numpy 数组。"""
        if not texts:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        if self._custom_embed is not None:
            vectors = await self._retry(lambda: self._custom_embed(texts), "embedding")
            return np.asarray(vectors, dtype=np.float32)

        all_vectors: list[list[float]] = []
        batch = self.config.embedding_batch_num
        for i in range(0, len(texts), batch):
            part = texts[i : i + batch]

            async def _do_batch(part: list[str] = part) -> Any:
                async with self._emb_sem:
                    return await self._emb_client.embeddings.create(  # type: ignore[union-attr]
                        model=self.config.embedding_model, input=part
                    )

            resp = await self._retry(_do_batch, "embedding")
            all_vectors.extend(d.embedding for d in resp.data)
        return np.asarray(all_vectors, dtype=np.float32)

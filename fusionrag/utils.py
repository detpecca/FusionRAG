"""通用工具: tokenizer、哈希 id、token 截断、并发限制。"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import re
from typing import Any, Callable, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Tokenizer (tiktoken, 失败时退化为按字符估算)
# ---------------------------------------------------------------------------


class Tokenizer:
    """轻量 tokenizer 封装, tiktoken 不可用时退化为字符级可逆编码。"""

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self._encoder = None
        try:
            import tiktoken

            self._encoder = tiktoken.encoding_for_model(model_name)
        except Exception:
            self._encoder = None

    def encode(self, text: str) -> list[int]:
        if self._encoder is not None:
            return self._encoder.encode(text)
        # 退化方案: 字符级编码 (1 token = 1 字符, 可逆)
        return [ord(c) for c in text]

    def decode(self, tokens: Iterable[int]) -> str:
        if self._encoder is not None:
            return self._encoder.decode(list(tokens))
        return "".join(chr(t) for t in tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


_DEFAULT_TOKENIZER: Tokenizer | None = None


def get_tokenizer() -> Tokenizer:
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = Tokenizer()
    return _DEFAULT_TOKENIZER


# ---------------------------------------------------------------------------
# 哈希 id (md5)
# ---------------------------------------------------------------------------


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def compute_args_hash(*args: Any) -> str:
    return hashlib.md5("|".join(str(a) for a in args).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------


def sanitize_text(text: str) -> str:
    """清理 LLM 输出: 去首尾空白/引号。"""
    if not text:
        return ""
    return text.strip().strip('"').strip("'")


def truncate_list_by_token_size(
    items: list[dict], key: Callable[[dict], str], max_token_size: int
) -> list[dict]:
    """按 key 提取的文本累计 token 数截断列表。"""
    tokenizer = get_tokenizer()
    if max_token_size <= 0:
        return []
    results = []
    tokens = 0
    for item in items:
        t = tokenizer.count(key(item))
        if tokens + t > max_token_size:
            break
        results.append(item)
        tokens += t
    return results


def merge_source_ids(existing: str, new_ids: Iterable[str], sep: str, limit: int) -> str:
    """合并 <SEP> 连接的 source_id 列表: 保序去重并截断 (KEEP 旧值优先)。"""
    seen: list[str] = []
    id_set: set[str] = set()
    for sid in (existing.split(sep) if existing else []) + list(new_ids):
        if sid and sid not in id_set:
            id_set.add(sid)
            seen.append(sid)
    return sep.join(seen[:limit])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (d,) 或 (n,d); b: (m,d) -> 相似度矩阵/向量。"""
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# 并发限制装饰器
# ---------------------------------------------------------------------------


def limit_async_func_call(max_concurrent: int) -> Callable:
    def decorator(func: Callable) -> Callable:
        semaphore = asyncio.Semaphore(max_concurrent)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def remove_think_tags(text: str) -> str:
    """移除推理模型的 <think>...</think> 标签。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

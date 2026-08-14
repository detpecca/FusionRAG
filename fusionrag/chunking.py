"""文档切分: 按 token 滑窗切分。"""

from __future__ import annotations

from .utils import Tokenizer


def chunking_by_token_size(
    content: str,
    tokenizer: Tokenizer,
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100,
) -> list[dict]:
    """将长文本按 token 数滑窗切分。

    返回: [{tokens, content, chunk_order_index}, ...]
    (字段精简: 仅保留检索与排序必需字段)
    """
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError("chunk_overlap_token_size must be smaller than chunk_token_size")

    tokens = tokenizer.encode(content)
    step = chunk_token_size - chunk_overlap_token_size
    results: list[dict] = []
    for index, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + chunk_token_size]
        if not window:
            continue
        results.append(
            {
                "tokens": len(window),
                "content": tokenizer.decode(window).strip(),
                "chunk_order_index": index,
            }
        )
    return results

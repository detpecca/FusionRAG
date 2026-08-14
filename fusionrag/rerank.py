"""Rerank 精排服务。

向量粗排 (余弦相似度) 噪声较大, rerank 模型对 query + 候选 chunk 做交叉编码
精排, 是检索质量性价比最高的一步改进。

接口约定 (按 base_url 自动选择):
- Jina/Cohere 风格 (默认): POST {base_url}/rerank
    请求 {"model","query","documents","top_n"} -> 响应 {"results":[...]}
  vLLM / Xinference / Jina / Cohere 均遵循该约定。
- 阿里云百炼 DashScope 原生 (base_url 含 dashscope.aliyuncs.com):
    POST {base_url} 原样 (完整服务路径)
    请求 {"model","input":{"query","documents"},"parameters":{...}}
    -> 响应 {"output":{"results":[...]}}
  配置示例: RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank

失败兜底: rerank 服务不可达/报错/未配置时返回 None, 调用方保留原始排序,
绝不因精排失败阻断问答主链路。
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import FusionRAGConfig

logger = logging.getLogger("fusionrag.rerank")


class RerankService:
    """统一 rerank 入口。config.rerank_func 可注入自定义实现(测试用),

    签名: async (query: str, documents: list[str]) -> list[tuple[int, float]]
    返回 (原文档下标, 相关分) 列表, 不要求有序。
    """

    def __init__(self, config: FusionRAGConfig) -> None:
        self.config = config
        self._custom = config.rerank_func

    @property
    def available(self) -> bool:
        return self._custom is not None or bool(self.config.rerank_model)

    async def rerank(
        self, query: str, documents: list[str]
    ) -> Optional[list[tuple[int, float]]]:
        """精排 documents, 返回 (index, score) 列表; 不可用/失败返回 None。"""
        if not documents or not self.available:
            return None
        if self._custom is not None:
            try:
                return list(await self._custom(query, documents))
            except Exception as e:
                logger.warning("rerank 调用失败, 回退原始排序: %s", e)
                return None
        try:
            return await self._http_rerank(query, documents)
        except Exception as e:
            logger.warning("rerank 服务调用失败, 回退原始排序: %s", e)
            return None

    async def _http_rerank(
        self, query: str, documents: list[str]
    ) -> list[tuple[int, float]]:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "httpx is required for the default rerank client "
                "(openai 依赖自带). 或注入自定义 rerank_func."
            ) from e

        base_url = (self.config.rerank_base_url or self.config.llm_base_url).rstrip("/")
        api_key = self.config.rerank_api_key or self.config.llm_api_key
        url, payload = _build_request(
            base_url, self.config.rerank_model, query, documents
        )
        async with httpx.AsyncClient(timeout=self.config.rerank_timeout) as client:
            resp = await client.post(
                url, json=payload, headers={"Authorization": f"Bearer {api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()
        return _parse_results(data)


def _is_dashscope(base_url: str) -> bool:
    return "dashscope.aliyuncs.com" in base_url


def _build_request(
    base_url: str, model: str, query: str, documents: list[str]
) -> tuple[str, dict]:
    """按服务风格构造 (url, payload): DashScope 原生 / Jina-Cohere 风格。"""
    if _is_dashscope(base_url):
        return base_url, {
            "model": model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": len(documents), "return_documents": False},
        }
    return f"{base_url}/rerank", {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
    }


def _parse_results(data: dict) -> list[tuple[int, float]]:
    """两种风格的响应统一解析为 [(index, score)]。"""
    results = data.get("results") or data.get("output", {}).get("results", [])
    return [(r["index"], float(r["relevance_score"])) for r in results]


async def apply_rerank(
    query: str,
    chunks: list[dict],
    service: Optional[RerankService],
    config: FusionRAGConfig,
) -> list[dict]:
    """对合并后的 chunk 列表精排: 重排 + 低分过滤 + top_n 截断。

    chunks 元素需含 "content"; 命中精排的条目附加 "rerank_score"。
    服务不可用或失败时原样返回 (顺序不变)。
    """
    if service is None or not chunks:
        return chunks
    scores = await service.rerank(query, [c["content"] for c in chunks])
    if scores is None:
        return chunks

    kept = [
        (i, s) for i, s in scores if 0 <= i < len(chunks) and s >= config.min_rerank_score
    ]
    kept.sort(key=lambda x: -x[1])
    reranked = [{**chunks[i], "rerank_score": s} for i, s in kept[: config.rerank_top_n]]
    logger.info(
        "rerank: %d 候选 -> %d 保留 (top_n=%d, min_score=%.2f)",
        len(chunks), len(reranked), config.rerank_top_n, config.min_rerank_score,
    )
    return reranked

"""三类存储的本地实现:

- JsonKVStorage        键值存储 (文档/chunk 原文/缓存等)
- SimpleVectorStorage  numpy 向量库 (chunk/实体/关系向量, 余弦相似度)
- NetworkXGraphStorage 知识图谱 (节点/边)

全部落盘到 working_dir 下的 JSON 文件, index_done_callback() 负责 flush。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional

import networkx as nx
import numpy as np

from .utils import cosine_similarity

logger = logging.getLogger("fusionrag.storage")


# ---------------------------------------------------------------------------
# 抽象存储接口 (可插拔后端: json / sqlite / redis ...)
#
# 业务代码 (core / merge / query / sessions) 只依赖这三个基类, 不依赖具体实现。
# 新增后端 = 继承对应基类 + 在下方工厂函数里加一个分支。
# 方法签名与现有 JSON 实现完全一致, 替换后端对调用方透明。
# ---------------------------------------------------------------------------


class BaseKVStorage(ABC):
    """命名空间级键值存储 (文档/chunk 原文/LLM 缓存/实体关系账本/会话)。"""

    @abstractmethod
    async def get_by_id(self, doc_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[Optional[dict]]: ...

    @abstractmethod
    async def keys(self) -> list[str]: ...

    @abstractmethod
    async def filter_keys(self, ids: set[str]) -> set[str]:
        """返回不存在于存储中的 id 集合。"""
        ...

    @abstractmethod
    async def upsert(self, data: dict[str, dict]) -> None: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    async def is_empty(self) -> bool: ...

    @abstractmethod
    async def index_done_callback(self) -> None:
        """把内存中的改动落盘 (flush)。"""
        ...


class BaseVectorStorage(ABC):
    """向量存储: content 入库时 embedding, query 返回按余弦相似度降序结果。"""

    @abstractmethod
    async def upsert(self, items: dict[str, dict]) -> None:
        """items: {id: {"content": str, "meta": {...}}}, content 会被 embedding。"""
        ...

    @abstractmethod
    async def query(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[dict]:
        """返回按余弦相似度降序的 [{id, content, distance, meta}]。"""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    async def get_by_id(self, doc_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def index_done_callback(self) -> None: ...


class BaseGraphStorage(ABC):
    """无向知识图谱: 节点=实体, 边=关系。"""

    @abstractmethod
    async def has_node(self, node_id: str) -> bool: ...

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]: ...

    @abstractmethod
    async def upsert_node(self, node_id: str, data: dict) -> None: ...

    @abstractmethod
    async def has_edge(self, src: str, tgt: str) -> bool: ...

    @abstractmethod
    async def get_edge(self, src: str, tgt: str) -> Optional[dict]: ...

    @abstractmethod
    async def get_edges_batch(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], dict]: ...

    @abstractmethod
    async def upsert_edge(self, src: str, tgt: str, data: dict) -> None: ...

    @abstractmethod
    async def node_degree(self, node_id: str) -> int: ...

    @abstractmethod
    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]: ...

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """批量取节点度数。默认逐个回退; 后端可覆盖为单次实现以减少锁竞争。"""
        return {n: await self.node_degree(n) for n in node_ids}

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """批量取各节点的关联边。默认逐个回退; 后端可覆盖为单次实现。"""
        return {n: await self.get_node_edges(n) for n in node_ids}

    @abstractmethod
    async def remove_node(self, node_id: str) -> None: ...

    @abstractmethod
    async def remove_edge(self, src: str, tgt: str) -> None: ...

    @abstractmethod
    async def index_done_callback(self) -> None: ...


def _atomic_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


class JsonKVStorage(BaseKVStorage):
    """namespace 级键值存储, 单 JSON 文件持久化。"""

    def __init__(self, namespace: str, working_dir: str) -> None:
        self.namespace = namespace
        self._path = os.path.join(working_dir, f"kv_{namespace}.json")
        self._lock = asyncio.Lock()
        self._data: dict[str, Any] = {}
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("KV 存储 %s 损坏, 已重置", self._path)
                self._data = {}

    async def get_by_id(self, doc_id: str) -> Optional[dict]:
        async with self._lock:
            return self._data.get(doc_id)

    async def get_by_ids(self, ids: list[str]) -> list[Optional[dict]]:
        async with self._lock:
            return [self._data.get(i) for i in ids]

    async def keys(self) -> list[str]:
        async with self._lock:
            return list(self._data.keys())

    async def filter_keys(self, ids: set[str]) -> set[str]:
        """返回不存在于存储中的 id 集合。"""
        async with self._lock:
            return {i for i in ids if i not in self._data}

    async def upsert(self, data: dict[str, dict]) -> None:
        async with self._lock:
            for k, v in data.items():
                self._data[k] = v

    async def delete(self, ids: list[str]) -> None:
        async with self._lock:
            for i in ids:
                self._data.pop(i, None)

    async def is_empty(self) -> bool:
        async with self._lock:
            return not self._data

    async def index_done_callback(self) -> None:
        async with self._lock:
            await asyncio.to_thread(_atomic_write_json, self._path, self._data)


class SimpleVectorStorage(BaseVectorStorage):
    """极简向量库: 内存 dict + 余弦相似度检索, JSON 持久化。"""

    def __init__(
        self,
        namespace: str,
        working_dir: str,
        embedding_func: Callable[[list[str]], Awaitable[np.ndarray]],
        embedding_dim: int,
    ) -> None:
        self.namespace = namespace
        self.embedding_dim = embedding_dim
        self._embed = embedding_func
        self._path = os.path.join(working_dir, f"vdb_{namespace}.json")
        self._lock = asyncio.Lock()
        self._data: dict[str, dict] = {}
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("向量库 %s 损坏, 已重置", self._path)
                self._data = {}

    async def upsert(self, items: dict[str, dict]) -> None:
        """items: {id: {"content": str, "meta": {...}}}, content 会被 embedding。"""
        if not items:
            return
        ids = list(items.keys())
        contents = [items[i]["content"] for i in ids]
        vectors = await self._embed(contents)
        async with self._lock:
            for idx, doc_id in enumerate(ids):
                self._data[doc_id] = {
                    "vector": vectors[idx].tolist(),
                    "content": contents[idx],
                    "meta": items[doc_id].get("meta", {}),
                }

    async def query(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[dict]:
        """返回按余弦相似度降序的 [{id, content, distance, meta}]。"""
        if query_embedding is None:
            if query is None:
                raise ValueError("query 与 query_embedding 至少提供一个")
            query_embedding = (await self._embed([query]))[0]
        async with self._lock:
            if not self._data:
                return []
            ids = list(self._data.keys())
            matrix = np.array([self._data[i]["vector"] for i in ids], dtype=np.float32)
            sims = cosine_similarity(query_embedding, matrix)[0]
            order = np.argsort(-sims)[:top_k]
            results = []
            for rank in order:
                score = float(sims[rank])
                if threshold is not None and score < threshold:
                    continue
                doc_id = ids[rank]
                results.append(
                    {
                        "id": doc_id,
                        "content": self._data[doc_id]["content"],
                        "distance": score,
                        "meta": self._data[doc_id].get("meta", {}),
                    }
                )
            return results

    async def delete(self, ids: list[str]) -> None:
        async with self._lock:
            for i in ids:
                self._data.pop(i, None)

    async def get_by_id(self, doc_id: str) -> Optional[dict]:
        async with self._lock:
            return self._data.get(doc_id)

    async def index_done_callback(self) -> None:
        async with self._lock:
            await asyncio.to_thread(_atomic_write_json, self._path, self._data)


class NetworkXGraphStorage(BaseGraphStorage):
    """无向知识图谱, NetworkX 实现, node-link JSON 持久化。"""

    def __init__(self, namespace: str, working_dir: str) -> None:
        self.namespace = namespace
        self._path = os.path.join(working_dir, f"graph_{namespace}.json")
        self._lock = asyncio.Lock()
        self._graph = nx.Graph()
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._graph = nx.node_link_graph(json.load(f), edges="edges")
            except (json.JSONDecodeError, OSError, ValueError):
                logger.warning("图谱 %s 损坏, 已重置", self._path)
                self._graph = nx.Graph()

    async def has_node(self, node_id: str) -> bool:
        async with self._lock:
            return self._graph.has_node(node_id)

    async def get_node(self, node_id: str) -> Optional[dict]:
        async with self._lock:
            if self._graph.has_node(node_id):
                return dict(self._graph.nodes[node_id])
            return None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        async with self._lock:
            return {
                n: dict(self._graph.nodes[n]) for n in node_ids if self._graph.has_node(n)
            }

    async def upsert_node(self, node_id: str, data: dict) -> None:
        async with self._lock:
            self._graph.add_node(node_id, **data)

    async def has_edge(self, src: str, tgt: str) -> bool:
        async with self._lock:
            return self._graph.has_edge(src, tgt)

    async def get_edge(self, src: str, tgt: str) -> Optional[dict]:
        async with self._lock:
            if self._graph.has_edge(src, tgt):
                return dict(self._graph.edges[src, tgt])
            return None

    async def get_edges_batch(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
        async with self._lock:
            return {
                (s, t): dict(self._graph.edges[s, t])
                for s, t in pairs
                if self._graph.has_edge(s, t)
            }

    async def upsert_edge(self, src: str, tgt: str, data: dict) -> None:
        async with self._lock:
            self._graph.add_edge(src, tgt, **data)

    async def node_degree(self, node_id: str) -> int:
        async with self._lock:
            return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        async with self._lock:
            if not self._graph.has_node(node_id):
                return []
            return [(s, t) for s, t in self._graph.edges(node_id)]

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """单次锁内批量取度数, 避免逐实体抢锁 (对齐 _local_retrieve 的批量需求)。"""
        async with self._lock:
            return {
                n: (self._graph.degree(n) if self._graph.has_node(n) else 0)
                for n in node_ids
            }

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """单次锁内批量取各节点关联边。"""
        async with self._lock:
            return {
                n: (
                    [(s, t) for s, t in self._graph.edges(n)]
                    if self._graph.has_node(n)
                    else []
                )
                for n in node_ids
            }

    async def remove_node(self, node_id: str) -> None:
        """删除节点 (NetworkX 会连带删除其全部关联边)。"""
        async with self._lock:
            if self._graph.has_node(node_id):
                self._graph.remove_node(node_id)

    async def remove_edge(self, src: str, tgt: str) -> None:
        async with self._lock:
            if self._graph.has_edge(src, tgt):
                self._graph.remove_edge(src, tgt)

    async def index_done_callback(self) -> None:
        async with self._lock:
            data = nx.node_link_data(self._graph, edges="edges")
            await asyncio.to_thread(_atomic_write_json, self._path, data)


# ---------------------------------------------------------------------------
# 存储工厂: 按 config.storage_backend 分发后端。
#
# 新增后端 (sqlite / redis ...) 只需:
#   1. 在对应模块实现 BaseKVStorage / BaseVectorStorage / BaseGraphStorage;
#   2. 在下面的分支里加一个 elif。
# 业务代码 (core) 只调工厂, 不感知具体实现。
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 存储工厂: 每类存储独立按 config 的对应字段分发后端。
#
# 后端解析优先级: <类>_backend > storage_backend > "json"
#   (由 config.resolve_kv_backend / resolve_vector_backend / resolve_graph_backend 决定)
# 新增后端 (sqlite / redis ...) 只需:
#   1. 在对应模块实现 BaseKVStorage / BaseVectorStorage / BaseGraphStorage;
#   2. 在下面的分支里加一个 elif。
# 业务代码 (core) 只调工厂, 不感知具体实现。
# ---------------------------------------------------------------------------


def _resolve(config: Any, resolver: str, attr: str, default: str = "json") -> str:
    """优先调用 config.resolve_*() (支持三后端拆分), 回退到旧的单一字段。"""
    fn = getattr(config, resolver, None)
    if callable(fn):
        return fn() or default
    return getattr(config, attr, None) or getattr(config, "storage_backend", default) or default


def create_kv_storage(namespace: str, working_dir: str, config: Any) -> BaseKVStorage:
    backend = _resolve(config, "resolve_kv_backend", "kv_backend")
    if backend == "json":
        return JsonKVStorage(namespace, working_dir)
    if backend == "sqlite":
        from .storage_sqlite import SqliteKVStorage  # 延迟 import, 默认 json 用户不加载

        return SqliteKVStorage(namespace, working_dir)
    raise ValueError(
        f"未知的 KV 存储后端 KV_BACKEND={backend!r}, 目前支持: 'json' / 'sqlite'"
    )


def create_vector_storage(
    namespace: str,
    working_dir: str,
    embedding_func: Callable[[list[str]], Awaitable[np.ndarray]],
    embedding_dim: int,
    config: Any,
) -> BaseVectorStorage:
    backend = _resolve(config, "resolve_vector_backend", "vector_backend")
    if backend == "json":
        return SimpleVectorStorage(namespace, working_dir, embedding_func, embedding_dim)
    raise ValueError(
        f"未知或暂不支持的向量存储后端 VECTOR_BACKEND={backend!r}, 目前支持: 'json'"
    )


def create_graph_storage(
    namespace: str, working_dir: str, config: Any
) -> BaseGraphStorage:
    backend = _resolve(config, "resolve_graph_backend", "graph_backend")
    if backend == "json":
        return NetworkXGraphStorage(namespace, working_dir)
    raise ValueError(
        f"未知或暂不支持的图存储后端 GRAPH_BACKEND={backend!r}, 目前支持: 'json'"
    )

"""PostgreSQL 全家桶后端: KV (表+JSONB) / 向量 (pgvector) / 图 (Apache AGE)。

三类存储共用一个 PG 实例 (POSTGRES_DSN), 也可按类独立指定 DSN
(PG_DSN_KV / PG_DSN_VECTOR / PG_DSN_GRAPH)。相对 JSON 后端的优势:
- 持久化由 WAL 保证, index_done_callback 为 no-op, 无全量重写写放大
- 多进程/多 worker 安全 (MVCC + 行锁), 不再受单实例锁限制
- 检索走 HNSW 索引, 突破 JSON 后端每查询全量建矩阵的天花板

需要: asyncpg (pip install asyncpg), PG 侧 vector 扩展 (向量) 与
AGE 扩展 (图)。建表/建图在首次连接时自动完成 (IF NOT EXISTS 语义)。

AGE 使用要点 (实现时验证过):
- cypher() 第三参必须是裸参数, 值为 agtype map (JSON 字符串)
- SET r += $map 不支持参数 map, 属性需展开为 SET r.k = $k
- 返回的 agtype 标量是 JSON 字面量 (字符串带引号), 统一 json.loads 解包
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

import numpy as np

from .storage import BaseGraphStorage, BaseKVStorage, BaseVectorStorage

logger = logging.getLogger("fusionrag.storage_pg")


# ---------------------------------------------------------------------------
# 连接池: 同一 DSN 复用; AGE 池单独缓存 (连接级 search_path 不同)
# ---------------------------------------------------------------------------


async def _init_plain_conn(conn: Any) -> None:
    # jsonb 自动编解码, 业务代码直接读写 dict
    await conn.set_type_codec(
        "jsonb", schema="pg_catalog",
        encoder=json.dumps, decoder=json.loads, format="text",
    )


async def _init_age_conn(conn: Any) -> None:
    await _init_plain_conn(conn)
    await conn.set_type_codec(
        "agtype", schema="ag_catalog", encoder=str, decoder=str, format="text"
    )
    await conn.execute("LOAD 'age'")
    await conn.execute('SET search_path = ag_catalog, "$user", public')


_POOLS: dict[tuple[str, bool], Any] = {}
_POOLS_LOCK = asyncio.Lock()


def _try_import_asyncpg() -> Any:
    try:
        import asyncpg
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PostgreSQL 后端需要 asyncpg: pip install asyncpg "
            "(或安装 requirements.txt 后自动具备)"
        ) from e
    return asyncpg


async def _noop_reset(conn: Any) -> None:
    """空 reset: asyncpg 默认在连接归还时执行 RESET ALL, 会清掉 init 设置的
    search_path / LOAD 'age'; 传自定义 reset 后仅回滚残留事务, 会话变量保留。"""
    return None


async def _ddl(conn: Any, *statements: str) -> None:
    """以事务级咨询锁串行化 DDL。

    CREATE TABLE IF NOT EXISTS / create_graph 的"不存在检查"本身有竞态:
    多个连接 (或多进程) 并发首次建同一对象时, 复合类型/索引创建会撞
    pg_type 唯一约束。咨询锁跨进程生效, 事务提交自动释放。
    """
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", "fusionrag_ddl")
        for stmt in statements:
            await conn.execute(stmt)


async def _get_pool(dsn: str, age: bool = False) -> Any:
    asyncpg = _try_import_asyncpg()
    key = (dsn, age)
    async with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = await asyncpg.create_pool(
                dsn, min_size=1, max_size=10,
                init=_init_age_conn if age else _init_plain_conn,
                reset=_noop_reset,
            )
            _POOLS[key] = pool
        return pool


async def close_all_pools() -> None:
    """关闭全部连接池 (测试/优雅退出用)。"""
    async with _POOLS_LOCK:
        for pool in _POOLS.values():
            await pool.close()
        _POOLS.clear()


# ---------------------------------------------------------------------------
# KV: 每命名空间一张表 public.kv_{namespace}(key TEXT PK, value JSONB)
# ---------------------------------------------------------------------------


class PostgresKVStorage(BaseKVStorage):
    def __init__(self, namespace: str, dsn: str) -> None:
        self.namespace = namespace
        self._dsn = dsn
        self._table = f"public.kv_{namespace}"
        self._ready = False

    async def _pool(self) -> Any:
        return await _get_pool(self._dsn, age=False)

    async def _ensure_table(self, conn: Any) -> None:
        if not self._ready:
            await _ddl(
                conn,
                f"CREATE TABLE IF NOT EXISTS {self._table} "
                "(key TEXT PRIMARY KEY, value JSONB NOT NULL)",
            )
            self._ready = True

    async def get_by_id(self, doc_id: str) -> Optional[dict]:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            v = await conn.fetchval(
                f"SELECT value FROM {self._table} WHERE key = $1", doc_id
            )
            return v

    async def get_by_ids(self, ids: list[str]) -> list[Optional[dict]]:
        if not ids:
            return []
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            rows = await conn.fetch(
                f"SELECT key, value FROM {self._table} WHERE key = ANY($1)", ids
            )
            by_key = {r["key"]: r["value"] for r in rows}
            return [by_key.get(i) for i in ids]

    async def keys(self) -> list[str]:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            rows = await conn.fetch(f"SELECT key FROM {self._table} ORDER BY key")
            return [r["key"] for r in rows]

    async def filter_keys(self, ids: set[str]) -> set[str]:
        if not ids:
            return set()
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            rows = await conn.fetch(
                f"SELECT key FROM {self._table} WHERE key = ANY($1)", list(ids)
            )
            existing = {r["key"] for r in rows}
            return ids - existing

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            async with conn.transaction():
                # executemany 复用预编译语句, 比逐条 execute 快一个量级
                await conn.executemany(
                    f"INSERT INTO {self._table} (key, value) VALUES ($1, $2) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    [(k, v) for k, v in data.items()],
                )

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            await conn.execute(
                f"DELETE FROM {self._table} WHERE key = ANY($1)", ids
            )

    async def is_empty(self) -> bool:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            return not await conn.fetchval(
                f"SELECT EXISTS (SELECT 1 FROM {self._table})"
            )

    async def index_done_callback(self) -> None:
        """PG 持久化由 WAL 保证, 无需显式落盘。"""


# ---------------------------------------------------------------------------
# 向量: public.vdb_{namespace}(id TEXT PK, content, meta JSONB, embedding vector(dim))
# ---------------------------------------------------------------------------


def _parse_vector(text: str) -> list[float]:
    """pgvector 文本格式 '[0.1,0.2,...]' -> list[float]。"""
    return json.loads(text)


class PgvectorStorage(BaseVectorStorage):
    def __init__(
        self,
        namespace: str,
        dsn: str,
        embedding_func: Callable[[list[str]], Awaitable[np.ndarray]],
        embedding_dim: int,
    ) -> None:
        self.namespace = namespace
        self.embedding_dim = embedding_dim
        self._embed = embedding_func
        self._dsn = dsn
        self._table = f"public.vdb_{namespace}"
        self._ready = False

    async def _pool(self) -> Any:
        return await _get_pool(self._dsn, age=False)

    async def _ensure_table(self, conn: Any) -> None:
        if not self._ready:
            await _ddl(
                conn,
                "CREATE EXTENSION IF NOT EXISTS vector",
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "id TEXT PRIMARY KEY, "
                "content TEXT NOT NULL, "
                "meta JSONB NOT NULL DEFAULT '{}', "
                f"embedding vector({self.embedding_dim}) NOT NULL)",
                # HNSW 余弦距离索引, 查询走 ORDER BY embedding <=> $q
                f"CREATE INDEX IF NOT EXISTS idx_{self.namespace}_hnsw "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops)",
            )
            self._ready = True

    async def upsert(self, items: dict[str, dict]) -> None:
        if not items:
            return
        ids = list(items.keys())
        contents = [items[i]["content"] for i in ids]
        vectors = await self._embed(contents)
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            async with conn.transaction():
                await conn.executemany(
                    f"INSERT INTO {self._table} (id, content, meta, embedding) "
                    "VALUES ($1, $2, $3, $4::vector) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "content = EXCLUDED.content, meta = EXCLUDED.meta, "
                    "embedding = EXCLUDED.embedding",
                    [
                        (
                            i,
                            items[i]["content"],
                            items[i].get("meta", {}),
                            "[" + ",".join(repr(float(x)) for x in vec) + "]",
                        )
                        for i, vec in zip(ids, vectors)
                    ],
                )

    async def query(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[dict]:
        if query_embedding is None:
            if query is None:
                raise ValueError("query 与 query_embedding 至少提供一个")
            query_embedding = (await self._embed([query]))[0]
        q = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"
        cond = ""
        args: list[Any] = [q, top_k]
        if threshold is not None:
            # <=> 是余弦距离 (0=相同, 2=相反), 相似度阈值转成距离上限
            cond = "WHERE 1 - (embedding <=> $1::vector) >= $3"
            args.append(float(threshold))
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            rows = await conn.fetch(
                f"SELECT id, content, meta, 1 - (embedding <=> $1::vector) AS score "
                f"FROM {self._table} {cond} "
                "ORDER BY embedding <=> $1::vector LIMIT $2",
                *args,
            )
            return [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "distance": float(r["score"]),
                    "meta": r["meta"],
                }
                for r in rows
            ]

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1)", ids
            )

    async def get_by_id(self, doc_id: str) -> Optional[dict]:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_table(conn)
            r = await conn.fetchrow(
                f"SELECT content, meta, embedding::text AS vec "
                f"FROM {self._table} WHERE id = $1",
                doc_id,
            )
            if r is None:
                return None
            return {
                "vector": _parse_vector(r["vec"]),
                "content": r["content"],
                "meta": r["meta"],
            }

    async def index_done_callback(self) -> None:
        """no-op, 同 PostgresKVStorage。"""


# ---------------------------------------------------------------------------
# 图: AGE, 图名 fusionrag_{namespace}, 节点 label Entity, 关系 label REL
# 存储键语义与 NetworkXGraphStorage 一致: 无向, 边端点排序后存一个方向,
# 读写一律用无向匹配 -[r:REL]-
# ---------------------------------------------------------------------------


def _agtype_str(v: Any) -> str:
    """agtype 标量转 python: JSON 字面量 (字符串带引号) -> 值。"""
    if v is None:
        return ""
    out = json.loads(v)
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


def _agtype_obj(v: Any) -> dict:
    if v is None:
        return {}
    return json.loads(v)


def _cypher_props(props: dict) -> tuple[str, dict]:
    """把属性 dict 展开为 SET 子句片段 + agtype 参数 map。

    AGE 的 SET r += $map 不接受参数 map, 必须逐属性 SET r.k = $k;
    属性名来自代码内部白名单字段 (entity_id/weight/...), 无注入面。
    """
    sets = ", ".join(f"n.{k} = ${k}" for k in props)
    return sets, dict(props)


class AGEGraphStorage(BaseGraphStorage):
    _LABEL = "Entity"
    _REL = "REL"

    def __init__(self, namespace: str, dsn: str) -> None:
        self.namespace = namespace
        self._dsn = dsn
        self._graph = f"fusionrag_{namespace}".replace("-", "_")
        self._ready = False

    async def _pool(self) -> Any:
        return await _get_pool(self._dsn, age=True)

    async def _ensure_graph(self, conn: Any) -> None:
        if self._ready:
            return
        # 咨询锁串行化建图, 消除 count-then-create 的并发竞态
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))", "fusionrag_ddl"
            )
            n = await conn.fetchval(
                "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = $1", self._graph
            )
            if not n:
                await conn.execute(
                    "SELECT ag_catalog.create_graph($1)", self._graph
                )
                # label 表 (Entity/REL) 是首次写入时懒创建的, 多连接并发
                # 首写会竞态撞 "already exists"; 建图后在锁内预创建
                for fn, label in (
                    ("ag_catalog.create_vlabel", self._LABEL),
                    ("ag_catalog.create_elabel", self._REL),
                ):
                    try:
                        await conn.execute(f"SELECT {fn}($1, $2)", self._graph, label)
                    except Exception as e:
                        if "already exists" not in str(e):
                            raise
        self._ready = True

    async def _fetch(self, sql: str, *args: Any) -> list[Any]:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_graph(conn)
            return await conn.fetch(sql, *args)

    async def _fetchval(self, sql: str, *args: Any) -> Any:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_graph(conn)
            return await conn.fetchval(sql, *args)

    async def _execute(self, sql: str, *args: Any) -> None:
        pool = await self._pool()
        async with pool.acquire() as conn:
            await self._ensure_graph(conn)
            await conn.execute(sql, *args)

    # ------------------------------------------------------------------ 节点

    async def has_node(self, node_id: str) -> bool:
        v = await self._fetchval(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (n {entity_id: $id}) RETURN count(n) $cq$, $1) AS (c agtype)",
            json.dumps({"id": node_id}),
        )
        return json.loads(v) > 0

    async def get_node(self, node_id: str) -> Optional[dict]:
        rows = await self._fetch(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (n {entity_id: $id}) RETURN properties(n) $cq$, $1) "
            "AS (p agtype)",
            json.dumps({"id": node_id}),
        )
        return _agtype_obj(rows[0]["p"]) if rows else None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "UNWIND $ids AS x MATCH (n {entity_id: x}) "
            "RETURN properties(n) $cq$, $1) AS (p agtype)",
            json.dumps({"ids": node_ids}),
        )
        return {_agtype_obj(r["p"]).get("entity_id", ""): _agtype_obj(r["p"]) for r in rows}

    async def upsert_node(self, node_id: str, data: dict) -> None:
        props = {"entity_id": node_id, **data}
        sets, params = _cypher_props(props)
        await self._execute(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            f"MERGE (n:{self._LABEL} {{entity_id: $id}}) SET {sets} "
            "RETURN 1 $cq$, $1) AS (x agtype)",
            json.dumps({"id": node_id, **params}),
        )

    # ------------------------------------------------------------------ 边

    async def has_edge(self, src: str, tgt: str) -> bool:
        # 无向匹配: 内部按排序方向存储, 但调用方可能以任意顺序传端点
        v = await self._fetchval(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (a {entity_id: $s})-[r]-(b {entity_id: $t}) "
            "RETURN count(r) $cq$, $1) AS (c agtype)",
            json.dumps({"s": src, "t": tgt}),
        )
        return json.loads(v) > 0

    async def get_edge(self, src: str, tgt: str) -> Optional[dict]:
        rows = await self._fetch(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (a {entity_id: $s})-[r]-(b {entity_id: $t}) "
            "RETURN properties(r) $cq$, $1) AS (p agtype)",
            json.dumps({"s": src, "t": tgt}),
        )
        return _agtype_obj(rows[0]["p"]) if rows else None

    async def get_edges_batch(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}
        rows = await self._fetch(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "UNWIND $pairs AS p "
            "MATCH (a {entity_id: p[0]})-[r]-(b {entity_id: p[1]}) "
            "RETURN p[0], p[1], properties(r) $cq$, $1) "
            "AS (s agtype, t agtype, e agtype)",
            json.dumps({"pairs": [list(p) for p in pairs]}),
        )
        out: dict[tuple[str, str], dict] = {}
        for r in rows:
            key = tuple(sorted((_agtype_str(r["s"]), _agtype_str(r["t"]))))
            out[key] = _agtype_obj(r["e"])
        return out

    async def upsert_edge(self, src: str, tgt: str, data: dict) -> None:
        # 端点排序规范化: AGE 是有向存储, 同一对端点以不同顺序 MERGE 会建出
        # 两条反向边; 统一按排序方向存储, 对外保持无向语义
        s, t = sorted((src, tgt))
        sets = ", ".join(f"r.{k} = ${k}" for k in data)
        await self._execute(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            f"MERGE (a:{self._LABEL} {{entity_id: $s}}) "
            f"MERGE (b:{self._LABEL} {{entity_id: $t}}) "
            f"MERGE (a)-[r:{self._REL}]->(b) SET {sets} "
            "RETURN 1 $cq$, $1) AS (x agtype)",
            json.dumps({"s": s, "t": t, **data}),
        )

    # ------------------------------------------------------------------ 度与邻接

    async def node_degree(self, node_id: str) -> int:
        v = await self._fetchval(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (n {entity_id: $id})-[r]-(m) RETURN count(m) $cq$, $1) "
            "AS (c agtype)",
            json.dumps({"id": node_id}),
        )
        return json.loads(v) if v is not None else 0

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        rows = await self._fetch(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (n {entity_id: $id})-[r]-(m) "
            "RETURN n.entity_id, m.entity_id $cq$, $1) "
            "AS (s agtype, t agtype)",
            json.dumps({"id": node_id}),
        )
        return sorted(
            {tuple(sorted((_agtype_str(r["s"]), _agtype_str(r["t"])))) for r in rows}
        )

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "UNWIND $ids AS x "
            "OPTIONAL MATCH (n {entity_id: x})-[r]-(m) "
            "RETURN x, count(m) $cq$, $1) AS (x agtype, c agtype)",
            json.dumps({"ids": node_ids}),
        )
        return {_agtype_str(r["x"]): json.loads(r["c"]) for r in rows}

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}
        rows = await self._fetch(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "UNWIND $ids AS x "
            "MATCH (n {entity_id: x})-[r]-(m) "
            "RETURN x, n.entity_id, m.entity_id $cq$, $1) "
            "AS (x agtype, s agtype, t agtype)",
            json.dumps({"ids": node_ids}),
        )
        out: dict[str, list[tuple[str, str]]] = {n: [] for n in node_ids}
        for r in rows:
            out.setdefault(_agtype_str(r["x"]), []).append(
                tuple(sorted((_agtype_str(r["s"]), _agtype_str(r["t"]))))
            )
        return {k: sorted(set(v)) for k, v in out.items()}

    # ------------------------------------------------------------------ 删除

    async def remove_node(self, node_id: str) -> None:
        await self._execute(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (n {entity_id: $id}) DETACH DELETE n $cq$, $1) AS (x agtype)",
            json.dumps({"id": node_id}),
        )

    async def remove_edge(self, src: str, tgt: str) -> None:
        await self._execute(
            f"SELECT * FROM cypher('{self._graph}', $cq$ "
            "MATCH (a {entity_id: $s})-[r]-(b {entity_id: $t}) DELETE r $cq$, $1) "
            "AS (x agtype)",
            json.dumps({"s": src, "t": tgt}),
        )

    async def index_done_callback(self) -> None:
        """no-op, 同 PostgresKVStorage。"""

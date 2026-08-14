"""基于标准库 sqlite3 的 KV 存储后端 (BaseKVStorage 实现)。

与 JsonKVStorage 的行为逐一等价, 只把落盘从"全量重写整个 JSON"换成
"增量写脏行", 消除写放大:

- 启动时从 SQLite 全量 load 进内存 self._data (读操作全走内存, 与 JSON 一致);
- upsert/delete 改内存 dict, 并把受影响 key 记入 _dirty / _deleted;
- index_done_callback 仅对脏行 INSERT OR REPLACE、对删除行 DELETE, 再 commit;
- get_by_id 返回内存对象引用, 保住 sessions.append_turn 的
  "取出 -> mutate -> upsert" 模式。

同步 sqlite3 调用统一用 asyncio.to_thread 包一层, 避免阻塞事件循环。
落盘文件 kv_{namespace}.db 与 JSON 后端的 kv_{namespace}.json 并列, 后端隔离可共存。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from typing import Any, Optional

from .storage import BaseKVStorage

logger = logging.getLogger("fusionrag.storage_sqlite")


class SqliteKVStorage(BaseKVStorage):
    """namespace 级键值存储, SQLite 单文件持久化 + 内存缓存 + 增量落盘。"""

    def __init__(self, namespace: str, working_dir: str) -> None:
        self.namespace = namespace
        self._path = os.path.join(working_dir, f"kv_{namespace}.db")
        self._lock = asyncio.Lock()
        self._data: dict[str, Any] = {}
        self._dirty: set[str] = set()
        self._deleted: set[str] = set()

        os.makedirs(working_dir, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (id TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()
        # 全量载入内存 (与 JsonKVStorage 一致: 读操作无需再碰磁盘)
        try:
            for key, value in self._conn.execute("SELECT id, value FROM kv"):
                try:
                    self._data[key] = json.loads(value)
                except json.JSONDecodeError:
                    logger.warning("KV(sqlite) %s 的键 %s 反序列化失败, 已跳过", self._path, key)
        except sqlite3.DatabaseError:
            logger.warning("KV 存储 %s 损坏, 已按空库处理", self._path)
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
                self._dirty.add(k)
                self._deleted.discard(k)

    async def delete(self, ids: list[str]) -> None:
        async with self._lock:
            for i in ids:
                if self._data.pop(i, None) is not None or i not in self._deleted:
                    self._deleted.add(i)
                    self._dirty.discard(i)

    async def is_empty(self) -> bool:
        async with self._lock:
            return not self._data

    async def index_done_callback(self) -> None:
        async with self._lock:
            if not self._dirty and not self._deleted:
                return
            # 快照后在 to_thread 里执行同步 sqlite3, 避免阻塞事件循环
            rows = [
                (k, json.dumps(self._data[k], ensure_ascii=False))
                for k in self._dirty
                if k in self._data
            ]
            deleted = list(self._deleted)
            await asyncio.to_thread(self._flush_sync, rows, deleted)
            self._dirty.clear()
            self._deleted.clear()

    def _flush_sync(self, rows: list[tuple[str, str]], deleted: list[str]) -> None:
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (id, value) VALUES (?, ?)", rows
            )
        if deleted:
            self._conn.executemany("DELETE FROM kv WHERE id = ?", [(d,) for d in deleted])
        self._conn.commit()

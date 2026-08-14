"""会话历史管理: JSON 持久化 + 滑动窗口。

长对话处理策略: 只保留最近 max_history_messages 条消息发给 LLM,
历史只参与对话上下文, 不参与检索。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from .storage import BaseKVStorage, create_kv_storage


class SessionStore:
    def __init__(
        self,
        working_dir: str,
        max_history_messages: int = 20,
        config: Any = None,
    ) -> None:
        self._kv: BaseKVStorage = create_kv_storage("chat_sessions", working_dir, config)
        self._max = max_history_messages
        self._lock = asyncio.Lock()

    async def get_history(self, session_id: str) -> list[dict]:
        """返回 OpenAI 风格 [{role, content}, ...], 已按滑动窗口截断。"""
        record = await self._kv.get_by_id(session_id)
        if not record:
            return []
        messages = record.get("messages", [])[-self._max :]
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    async def append_turn(self, session_id: str, user_content: str, assistant_content: str) -> None:
        async with self._lock:
            record = await self._kv.get_by_id(session_id) or {"messages": []}
            now = time.time()
            record["messages"].append(
                {"role": "user", "content": user_content, "created_at": now}
            )
            record["messages"].append(
                {"role": "assistant", "content": assistant_content, "created_at": now}
            )
            record["messages"] = record["messages"][-self._max :]
            record["updated_at"] = now
            await self._kv.upsert({session_id: record})
            await self._kv.index_done_callback()

    async def clear(self, session_id: str) -> bool:
        record = await self._kv.get_by_id(session_id)
        if record is None:
            return False
        await self._kv.delete([session_id])
        await self._kv.index_done_callback()
        return True

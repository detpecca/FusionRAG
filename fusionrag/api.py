"""FastAPI 服务层: 文档导入 / 对话问答 / 会话管理。

| 方法   | 路径                               | 说明                       |
| ------ | ---------------------------------- | -------------------------- |
| POST   | /api/v1/document/import            | 提交文档内容               |
| POST   | /api/v1/chat/completions           | 发起对话(支持流式返回)     |
| GET    | /api/v1/chat/history/{sessionId}   | 查看当前会话的上下文记录   |
| DELETE | /api/v1/chat/session/{sessionId}   | 重置/清理会话              |

统一响应结构: {"code": 0, "message": "success", "data": {...}}; 业务错误 code != 0。
流式对话使用 SSE (text/event-stream), 每帧 `data: {"session_id":..., "delta": "..."}`, 结束帧 `data: [DONE]`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import FusionRAGConfig
from .core import FusionRAG
from .query import QUERY_MODES, QueryParam
from .utils import compute_mdhash_id

logger = logging.getLogger("fusionrag.api")


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------


class DocumentImportRequest(BaseModel):
    document: str = Field(..., min_length=1, description="Markdown 或纯文本长文本")
    doc_id: Optional[str] = Field(None, description="可选文档 id, 缺省按内容哈希生成")
    wait: bool = Field(
        True,
        description="True 同步等待导入完成; False 立即返回 PROCESSING, 后台执行, "
        "用 GET /api/v1/document/{doc_id} 轮询状态",
    )


class ChatCompletionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 id")
    message: str = Field(..., min_length=1, description="用户问题")
    mode: str = Field("hybrid", description="检索模式: local/global/hybrid/naive")
    stream: bool = Field(False, description="是否流式返回(SSE)")
    top_k: Optional[int] = Field(None, description="图谱检索 top_k, 缺省用全局配置")
    enable_rerank: Optional[bool] = Field(
        None, description="是否启用 rerank 精排, 缺省跟随服务端配置(需配置 RERANK_MODEL)"
    )
    include_references: bool = Field(True, description="是否在响应中附带引用列表")


def _ok(data: dict) -> dict:
    return {"code": 0, "message": "success", "data": data}


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------


def create_app(rag: Optional[FusionRAG] = None) -> FastAPI:
    rag = rag or FusionRAG(FusionRAGConfig.from_env())
    app = FastAPI(title="FusionRAG", version="0.1.0")
    app.state.rag = rag
    app.state.import_tasks: set[asyncio.Task] = set()  # 后台导入任务引用

    webui_index = Path(__file__).resolve().parent.parent / "webui" / "index.html"

    # 可选 API Key 鉴权: 设置环境变量 FUSIONRAG_API_KEY 后, /api/* 端点
    # 需要 Authorization: Bearer <key> 或 X-API-Key 头; 未设置则不启用
    api_key = os.environ.get("FUSIONRAG_API_KEY")

    if api_key:
        @app.middleware("http")
        async def check_api_key(request: Request, call_next):
            if request.url.path.startswith("/api/"):
                auth = request.headers.get("Authorization", "")
                token = auth.removeprefix("Bearer ").strip() or request.headers.get(
                    "X-API-Key", ""
                )
                if token != api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"code": 401, "message": "invalid or missing API key", "data": None},
                    )
            return await call_next(request)

    @app.get("/", include_in_schema=False)
    async def webui():
        return FileResponse(webui_index)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # 内部异常细节只进日志, 不回传客户端 (避免泄漏路径/配置等敏感信息)
        logger.exception("请求处理失败: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "internal server error", "data": None},
        )

    # ---------------------------------------------------------- 文档导入
    @app.post("/api/v1/document/import")
    async def import_document(req: DocumentImportRequest):
        if not req.wait:
            # 异步导入: 立即返回 PROCESSING, 后台跑流水线, 状态可轮询
            doc_id = req.doc_id or compute_mdhash_id(req.document, prefix="doc-")

            def _on_done(task: asyncio.Task) -> None:
                app.state.import_tasks.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    logger.error("后台导入 %s 失败: %s", doc_id, task.exception())

            task = asyncio.create_task(
                rag.ainsert(req.document, doc_id=req.doc_id)
            )
            task.add_done_callback(_on_done)
            app.state.import_tasks.add(task)  # 持引用防 GC
            return _ok({"doc_id": doc_id, "status": "PROCESSING", "async": True})
        try:
            result = await rag.ainsert(req.document, doc_id=req.doc_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _ok(result)

    # ---------------------------------------------------------- 文档状态
    @app.get("/api/v1/document/{doc_id}")
    async def get_document(doc_id: str):
        """查询单个文档的导入状态 (异步导入的轮询端点)。"""
        rec = await rag.full_docs.get_by_id(doc_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
        return _ok(
            {
                "doc_id": doc_id,
                "title": rec.get("title") or doc_id,
                "status": rec.get("status", "UNKNOWN"),
                "chunks": rec.get("chunks_count", 0),
                "entities": rec.get("entities_count", 0),
                "relations": rec.get("relations_count", 0),
                "error_msg": rec.get("error_msg"),
                "created_at": rec.get("created_at", 0),
                "updated_at": rec.get("updated_at", 0),
            }
        )

    # ---------------------------------------------------------- 文档列表
    @app.get("/api/v1/documents")
    async def list_documents():
        docs = await rag.alist_documents()
        return _ok({"documents": docs, "total": len(docs)})

    # ---------------------------------------------------------- 删除文档
    @app.delete("/api/v1/document/{doc_id}")
    async def delete_document(doc_id: str):
        """删除文档并保持图谱一致: 无源实体/关系真删, 共享的扣减来源与权重。"""
        try:
            stats = await rag.adelete_by_doc_id(doc_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return _ok(stats)

    # ------------------------------------------------- 文件导入 (SSE 进度)
    ALLOWED_EXTS = {".md", ".markdown", ".txt"}

    @app.post("/api/v1/document/import-file")
    async def import_document_file(
        file: Optional[UploadFile] = File(None),
        text: Optional[str] = Form(None),
        doc_id: Optional[str] = Form(None),
    ):
        """上传 .md/.txt 文件 (或直接提交 text 字段), SSE 流式返回导入进度。

        进度事件: {"stage": chunked|vectorized|extracting|merging|finalizing, "current": n, "total": n}
        结束事件: {"stage": "done", "result": {...}} 或 {"stage": "error", "message": "..."}
        """
        title: Optional[str] = None
        if file is not None and file.filename:
            filename = file.filename
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in ALLOWED_EXTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"仅支持 {sorted(ALLOWED_EXTS)} 格式, 收到: {filename}",
                )
            raw = await file.read()
            content = ""
            for enc in ("utf-8", "gb18030"):  # 兼容中文 Windows 文本
                try:
                    content = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            title = filename
        elif text and text.strip():
            content = text
        else:
            raise HTTPException(status_code=400, detail="file 与 text 至少提供一个")
        if not content.strip():
            raise HTTPException(status_code=400, detail="文档内容不能为空")

        async def sse_generator():
            queue: asyncio.Queue = asyncio.Queue()

            def cb(stage: str, current: int, total: int) -> None:
                queue.put_nowait({"stage": stage, "current": current, "total": total})

            task = asyncio.create_task(
                rag.ainsert(
                    content, doc_id=doc_id or None, title=title, progress_callback=cb
                )
            )
            while not (task.done() and queue.empty()):
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # 保活
            exc = task.exception()
            if exc is not None:
                payload = {"stage": "error", "message": str(exc)}
            else:
                payload = {"stage": "done", "result": task.result()}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # ---------------------------------------------------------- 对话
    @app.post("/api/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if req.mode not in QUERY_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"mode 必须是 {list(QUERY_MODES)} 之一",
            )
        history = await rag.sessions.get_history(req.session_id)
        param = QueryParam(
            mode=req.mode,
            stream=req.stream,
            conversation_history=history,
            top_k=req.top_k or rag.config.top_k,
            chunk_top_k=rag.config.chunk_top_k,
            max_entity_tokens=rag.config.max_entity_tokens,
            max_relation_tokens=rag.config.max_relation_tokens,
            max_total_tokens=rag.config.max_total_tokens,
            enable_rerank=req.enable_rerank if req.enable_rerank is not None else True,
            include_references=req.include_references,
        )
        result = await rag.aquery(req.message, param)

        if not req.stream:
            answer = result.content or ""
            await rag.sessions.append_turn(req.session_id, req.message, answer)
            return _ok(
                {
                    "session_id": req.session_id,
                    "answer": answer,
                    "mode": req.mode,
                    "references": result.raw_data.get("references", []),
                    "retrieval": result.raw_data,
                }
            )

        async def sse_generator():
            collected: list[str] = []
            try:
                async for delta in result.response_iterator:  # type: ignore[union-attr]
                    collected.append(delta)
                    yield f"data: {json.dumps({'session_id': req.session_id, 'delta': delta}, ensure_ascii=False)}\n\n"
                # 引用帧: 流式正文结束后、结束前发送
                if req.include_references:
                    refs = result.raw_data.get("references", [])
                    yield f"data: {json.dumps({'session_id': req.session_id, 'references': refs}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception("流式生成失败")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            finally:
                # 客户端中途断连 (GeneratorExit/CancelledError) 也把已生成的
                # 部分答案落进会话历史, 否则该轮对话丢失、上下文断裂
                if collected:
                    try:
                        await rag.sessions.append_turn(
                            req.session_id, req.message, "".join(collected)
                        )
                    except Exception:
                        logger.exception("流式会话历史落盘失败")

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # ---------------------------------------------------------- 会话历史
    @app.get("/api/v1/chat/history/{session_id}")
    async def get_history(session_id: str):
        messages = await rag.sessions.get_history(session_id)
        return _ok({"session_id": session_id, "messages": messages})

    # ---------------------------------------------------------- 清理会话
    @app.delete("/api/v1/chat/session/{session_id}")
    async def clear_session(session_id: str):
        cleared = await rag.sessions.clear(session_id)
        return _ok({"session_id": session_id, "cleared": cleared})

    return app

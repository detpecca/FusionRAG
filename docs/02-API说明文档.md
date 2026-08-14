# API 说明文档

Base URL：`http://localhost:8000`
统一约定：非流式接口返回 `{"code": 0, "message": "success", "data": {...}}`；
参数校验失败返回 HTTP 422（FastAPI 默认结构）；业务错误返回 HTTP 4xx/5xx，`message`/`detail` 含原因。

服务启动后可访问 `GET /docs` 查看自动生成的 Swagger 文档，`GET /` 打开内置聊天界面。

---

## 1. 导入文档 `POST /api/v1/document/import`

提交一段长文本（Markdown 或 Txt），执行 切分 → 向量化 → 实体/关系抽取 → 去重合并入库。
相同内容（相同 doc_id）重复导入幂等跳过。

### 请求体（application/json）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `document` | string | 是 | Markdown / Txt 长文本，非空 |
| `doc_id` | string \| null | 否 | 自定义文档 id；缺省按内容 MD5 生成（`doc-<md5>`） |

### 响应 `data`

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | string | 文档 id |
| `status` | string | `PROCESSED` = 处理完成；`SKIPPED` = 重复导入跳过（失败为 `FAILED`，见 2.5 列表接口） |
| `chunks` | int | 切分出的文本片段数 |
| `entities` | int | 本次抽取合并的实体数 |
| `relations` | int | 本次抽取合并的关系数 |

### 示例

```bash
curl -X POST http://localhost:8000/api/v1/document/import \
  -H "Content-Type: application/json" \
  -d '{"document": "FusionRAG 是一个 GraphRAG 系统……"}'
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "doc_id": "doc-6438c6294a64efbc1f65f60dfcc0045d",
    "status": "PROCESSED",
    "chunks": 1,
    "entities": 32,
    "relations": 45
  }
}
```

---

## 2. 发起对话 `POST /api/v1/chat/completions`

基于知识库回答用户问题。相同 `session_id` 自动携带多轮历史（最近 20 条滑动窗口）。

### 请求体（application/json）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `session_id` | string | 是 | 会话 id，非空 |
| `message` | string | 是 | 用户问题，非空 |
| `mode` | string | 否 | 检索模式：`hybrid`（默认，图谱+向量三路混合）/ `local`（实体侧）/ `global`（关系侧）/ `naive`（纯向量）；其他值返回 400 |
| `stream` | bool | 否 | `true` 时 SSE 流式返回，默认 `false` |
| `top_k` | int \| null | 否 | 图谱检索 top_k，缺省用全局配置（40） |
| `enable_rerank` | bool \| null | 否 | 是否启用 rerank 精排，缺省 `true`；需服务端配置 `RERANK_MODEL` 才实际生效，未配置或服务不可用时自动回退原始排序 |
| `include_references` | bool | 否 | 是否附带引用列表，默认 `true` |

### 非流式响应 `data`（`stream=false`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 会话 id |
| `answer` | string | 完整回答（Markdown） |
| `mode` | string | 实际使用的检索模式 |
| `references` | array | 引用列表，每个元素对应一个进入上下文的片段：`{reference_id, chunk_id, doc_id, title, score, excerpt}`（`title` 为文档标题，`excerpt` 为片段前 120 字摘录，`score` 为 rerank 分或向量相似度） |
| `retrieval` | object | 检索过程数据：`{mode, keywords: {high_level, low_level}, entities: [...], relations: [[src,tgt],...], chunks: [...], references: [...]}` |

### 流式响应（`stream=true`）

`Content-Type: text/event-stream`，每帧一个增量；正文结束后、结束帧之前发送一帧引用列表
（`include_references=true` 时），最后一帧为 `data: [DONE]`；
流式结束后服务端自动把本轮问答写入会话历史。

```
data: {"session_id": "s1", "delta": "FusionRAG "}
data: {"session_id": "s1", "delta": "是一个……"}
data: {"session_id": "s1", "references": [{"reference_id": "1", "chunk_id": "chunk-...", "doc_id": "doc-...", "title": "示例文档.md", "score": 0.83, "excerpt": "……"}]}
data: [DONE]
```

错误帧：`data: {"error": "..."}`

### 示例

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "系统必须实现哪几个接口？", "mode": "hybrid"}'
```

---

## 3. 查看会话历史 `GET /api/v1/chat/history/{sessionId}`

### 路径参数

| 字段 | 类型 | 说明 |
|---|---|---|
| `sessionId` | string | 会话 id |

### 响应 `data`

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 会话 id |
| `messages` | array | `[{"role": "user" \| "assistant", "content": string}]`，按时间升序；最多返回最近 20 条（滑动窗口） |

### 示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "session_id": "s1",
    "messages": [
      {"role": "user", "content": "云脑平台有哪些版本迭代？"},
      {"role": "assistant", "content": "云脑平台于 2021 年发布 1.0，2024 年推出 3.0……"}
    ]
  }
}
```

---

## 4. 清理会话 `DELETE /api/v1/chat/session/{sessionId}`

### 路径参数

| 字段 | 类型 | 说明 |
|---|---|---|
| `sessionId` | string | 会话 id |

### 响应 `data`

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_id` | string | 会话 id |
| `cleared` | bool | `true` = 已清除；`false` = 该会话本就不存在 |

---

## 附：辅助接口（文件上传、进度展示与文档管理）

### 5. 文档列表 `GET /api/v1/documents`

响应 `data`：`{total: int, documents: [...]}`，按创建时间倒序。
`documents[]` 元素：

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | string | 文档 id |
| `title` | string | 标题（文件名或首行截取） |
| `status` | string | `PROCESSED` / `FAILED` / `PROCESSING` |
| `chunks` / `entities` / `relations` | int | 统计信息 |
| `content_length` | int | 原文长度（字符） |
| `error_msg` | string \| null | 失败原因（仅 FAILED 时非空） |
| `created_at` | float | 创建时间戳 |

### 6. 文件导入（带进度）`POST /api/v1/document/import-file`

`multipart/form-data`：`file`（.md/.markdown/.txt 文件，与 `text` 二选一）、`text`（文本，二选一）、`doc_id`（可选）。
以 SSE 返回导入进度：

```
data: {"stage": "chunked",    "current": 2, "total": 2}
data: {"stage": "vectorized", "current": 2, "total": 2}
data: {"stage": "extracting", "current": 1, "total": 2}
data: {"stage": "merging",    "current": 90, "total": 169}
data: {"stage": "finalizing", "current": 1, "total": 1}
data: {"stage": "done", "result": {"doc_id": "...", "status": "PROCESSED", "chunks": 2, "entities": 79, "relations": 90}}
```

失败时：`data: {"stage": "error", "message": "..."}`

### 7. 删除文档 `DELETE /api/v1/document/{doc_id}`

删除文档并保持图谱一致（策略见 `docs/01-架构设计说明.md` §5）：无源的实体/关系连同
图节点/边、向量、账本一并删除；被其他文档共享的仅扣减 `source_id` 并重算 `weight`。

### 路径参数

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | string | 文档 id；不存在返回 404 |

### 响应 `data`

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | string | 文档 id |
| `status` | string | 固定 `DELETED` |
| `chunks_deleted` | int | 删除的片段数 |
| `entities_deleted` / `entities_updated` | int | 真删 / 扣减更新的实体数 |
| `relations_deleted` / `relations_updated` | int | 真删 / 扣减更新的关系数 |

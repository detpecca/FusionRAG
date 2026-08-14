# FusionRAG —— 图增强检索生成（GraphRAG）企业知识库问答系统

把企业文档变成**可检索、可溯源、可治理**的知识：在向量检索之上构建实体/关系知识图谱，
以三路混合检索 + rerank 精排定位证据，让 LLM 严格基于上下文作答，并为每个答案附上引用来源。

零外部服务依赖即可运行（JSON + numpy + NetworkX 本地落盘），单容器 Docker 部署，
任何 OpenAI 兼容的 LLM / Embedding 端点均可接入。

## 特性

- **知识图谱构建**：实体/关系分隔符抽取 + gleaning 补抽 + 按名去重增量合并
  （描述 `<SEP>` 拼接、超阈值 LLM 摘要、weight 防重复累加、source_id 溯源）
- **三路混合检索**：`local`（实体侧）/ `global`（关系侧）/ `naive`（纯向量）/ `hybrid`（三路融合），
  round-robin 合并 + 动态 token 预算截断
- **Rerank 精排**：可选的二次精排（Jina/Cohere 风格 + 阿里云百炼 DashScope），
  低分过滤 + top_n 截断；服务不可用时自动回退原始排序，不阻断问答
- **引用溯源**：每个答案附带引用列表（文档标题 / 片段摘录 / 相关分），前端可折叠展示参考来源
- **文档全生命周期管理**：`.md/.txt` 文件上传（SSE 实时进度条）、文档列表、
  删除时按 chunk 账本三分支裁决（真删/扣减/不动），存活实体/关系的描述自动重建，全程幂等
- **多轮会话**：滑动窗口历史（只进对话、不进检索），SSE 流式输出
- **工程健壮**：LLM 结果全量缓存、429/5xx 指数退避重试、批量向量化、
  embedding 指纹校验（换模型/维度时拒绝启动，防向量静默损坏）
- **可插拔存储**：KV / 向量 / 图三类抽象接口，内置 JSON（零依赖）与 SQLite KV（增量落盘）后端
- **开箱即用**：ChatGPT 风格单文件 WebUI、Swagger 文档（`/docs`）、74 个测试用例

## 快速开始

### Docker（推荐）

```bash
cp .env.example .env      # 填入 LLM_API_KEY 等, 任何 OpenAI 兼容端点均可
docker compose up -d --build
```

打开 http://localhost:8000 即可开始对话与导入文档；知识库数据持久化在 `fusionrag_data` volume。

### 本地运行

```bash
pip install -r requirements.txt
cp .env.example .env
python -m fusionrag       # http://localhost:8000
```

## 架构

```
┌────────────────────────── 写入链路 (ainsert) ──────────────────────────┐
│ 长文本(.md/.txt) ─► token 滑窗切分 ─► chunk 原文入 KV / chunk 向量入向量库 │
│                          │                                            │
│                          └─► LLM 实体/关系抽取 (含 gleaning 补抽)        │
│                               ─► 去重合并 (描述 <SEP> 拼接/LLM 摘要、    │
│                                   weight 防重复累加、source_id 归并)    │
│                               ─► 知识图谱 (NetworkX) + 实体/关系向量库  │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────── 查询链路 (aquery) ───────────────────────────┐
│ 问题 ─► LLM 提取高/低层关键词                                          │
│          ├─ local : 低层关键词 → 实体向量库 top_k → 一跳关系+关联片段   │
│          ├─ global: 高层关键词 → 关系向量库 top_k → 相连实体+关联片段   │
│          └─ naive : 问题向量   → 片段向量库 top_k                       │
│        ─► 三路 round-robin 合并去重 ─► rerank 精排 (可选, 失败自动回退) │
│        ─► token 预算截断 ─► 组装上下文 ─► LLM 生成 (SSE 流式 / 多轮历史) │
│        ─► 引用列表 (chunk → 文档标题/摘录) 随响应一并返回               │
└────────────────────────────────────────────────────────────────────────┘
```

技术栈：Python 3.11+ / FastAPI / OpenAI 兼容 LLM 与 Embedding / NetworkX / numpy / tiktoken。

## 存储设计

全部持久化在 `WORKING_DIR` 下（默认 JSON 原子写，KV 可切换 SQLite），零外部服务依赖：
通过 `KV_BACKEND` / `VECTOR_BACKEND` / `GRAPH_BACKEND` 可独立替换后端。

| 存储 | 类型 | 用途 |
|---|---|---|
| `kv_full_docs` | KV | 文档原文与生命周期状态（PROCESSING/PROCESSED/FAILED）、标题与统计；按内容哈希幂等 |
| `kv_text_chunks` | KV | chunk 原文；检索命中后取回文本 |
| `vdb_chunks` | 向量库 | chunk 向量；naive 检索与 hybrid 的片段来源 |
| `kv_llm_cache` | KV | 抽取/摘要/关键词/问答的 LLM 结果缓存；重跑与删除重建近乎零成本 |
| `kv_entity_chunks` / `kv_relation_chunks` | KV | 实体/关系 → chunk 完整账本；增量合并与删除级联清理的依据 |
| `graph_chunk_entity_relation` | 图 | 知识图谱：实体节点 + 关系边；local/global 检索的一跳扩展 |
| `vdb_entities` / `vdb_relationships` | 向量库 | 实体/关系向量；local / global 检索入口 |
| `kv_chat_sessions` | KV | 多轮对话历史（滑动窗口） |

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/document/import` | 导入文档（文本） |
| POST | `/api/v1/document/import-file` | 上传 .md/.txt 文件（SSE 进度） |
| GET | `/api/v1/documents` | 文档列表 |
| DELETE | `/api/v1/document/{doc_id}` | 删除文档（图谱一致性维护） |
| POST | `/api/v1/chat/completions` | 对话问答（支持流式 / rerank / 引用） |
| GET | `/api/v1/chat/history/{sessionId}` | 会话历史 |
| DELETE | `/api/v1/chat/session/{sessionId}` | 清理会话 |

字段定义详见 [docs/02-API说明文档.md](docs/02-API说明文档.md)，交互式文档见 `/docs`。

## 配置

全部配置经环境变量注入（`.env`），常用项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` | — | 主模型（OpenAI 兼容端点） |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | — | 向量模型；缺省复用 LLM 的 key/endpoint |
| `RERANK_MODEL` / `RERANK_BASE_URL` | 空 | 精排服务（可选；不配则不启用） |
| `CHUNK_TOKEN_SIZE` / `CHUNK_OVERLAP_TOKEN_SIZE` | 1200 / 100 | 切分窗口与重叠 |
| `TOP_K` / `CHUNK_TOP_K` | 40 / 20 | 检索广度 |
| `MAX_GLEANING` | 1 | 补抽轮数 |
| `DELETE_REBUILD_DESCRIPTIONS` | true | 删除文档时重建存活实体/关系描述 |
| `KV_BACKEND` | json | KV 后端：`json` / `sqlite` |

完整清单见 [.env.example](.env.example)。

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q                        # 74 passed, 确定性 Fake LLM/Embedding/Rerank, 无需外部 API
python scripts/e2e_real.py       # 真实端点端到端验证 (需配置 .env 并启动服务)
python scripts/verify_delete.py  # 真实端点删除一致性验证
```

用例清单见 [docs/04-自测说明.md](docs/04-自测说明.md)。

## 路线图

FusionRAG 的下一步围绕"活的知识"展开——知识不止要被检索，还要被度量和治理：

- **时态知识图谱**：关系携带生效区间（valid_from / valid_to），支持"截至某时点"的查询
- **知识治理**：合并时的冲突检测与人工裁决工作流，矛盾事实不再被静默拼接
- **对话反哺**：从问答会话中挖掘候选事实，经确认后回流图谱，知识库越用越准
- **评估与反馈**：内置黄金集评估与答案反馈闭环，检索质量可度量、可回归
- **社区摘要与多跳推理**：面向全局概括与复杂对比问题的推理式查询

## 文档

- [docs/01-架构设计说明.md](docs/01-架构设计说明.md) — 切片策略、存储设计、长对话、删除一致性
- [docs/02-API说明文档.md](docs/02-API说明文档.md) — 接口字段定义
- [docs/03-核心源代码导读.md](docs/03-核心源代码导读.md) — 模块职责与调用顺序
- [docs/04-自测说明.md](docs/04-自测说明.md) — 测试用例清单

## 致谢

本项目深受 [LightRAG](https://github.com/HKUDS/LightRAG)（HKUDS）及其论文
*LightRAG: Simple and Fast Retrieval-Augmented Generation*（arXiv:2410.05779）的启发，
在此向作者与开源社区表示感谢。

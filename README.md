# Hexa-Ops — AIOps 智能运维平台

基于 **LangGraph + RAG + HITL** 的 Python 版智能运维 Agent 系统，支持告警诊断、知识库问答、人机协同审批，前端一体化交互。

---

## 目录

- [功能概览](#功能概览)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [核心模块说明](#核心模块说明)
- [快速启动](#快速启动)
- [环境变量配置](#环境变量配置)
- [API 接口文档](#api-接口文档)
- [运行流程详解](#运行流程详解)
- [RAG 评估](#rag-评估)
- [观测与追踪](#观测与追踪)

---

## 功能概览

| 功能模块 | 说明 |
|----------|------|
| **智能对话（Chat Agent）** | 多轮对话，支持流式输出（SSE），ReAct 推理框架，工具调用 |
| **AI Ops 告警诊断** | 输入告警信息 → 日志分析 + 指标采集 + RAG 知识检索 → 生成 RCA 诊断报告 |
| **人机协同审批（HITL）** | 高危操作（重启/回滚/DDL 等）自动暂停，前端弹出确认框，人工批准后再执行 |
| **混合 RAG 知识库** | FAISS 向量检索 + BM25 关键词检索，RRF 倒数秩融合，支持本地 Markdown 文档索引 |
| **流式响应** | 基于 SSE（Server-Sent Events），对话 token 逐字推送 |
| **LangFuse 可观测性** | 可选接入 LangFuse，追踪每条 LangGraph 调用链路 |
| **RAG 评估 Notebook** | Precision@K / Recall@K / MRR 三路对比（Hybrid vs FAISS vs BM25）|

---

## 系统架构

```
┌─────────────────────────────────────────┐
│               前端 (Vanilla JS)          │
│  ┌─────────────┐   ┌──────────────────┐ │
│  │  对话界面    │   │  AI Ops 诊断面板  │ │
│  │  (SSE 流式) │   │  (HITL 审批弹框) │ │
│  └──────┬──────┘   └────────┬─────────┘ │
└─────────┼────────────────────┼───────────┘
          │ HTTP/SSE           │ HTTP
          ▼                    ▼
┌─────────────────────────────────────────┐
│            FastAPI 后端 (port 8001)      │
│  /chat  /chat/stream  /ops/diagnose     │
│  /ops/approve  /knowledge/index         │
└──────┬──────────────────┬───────────────┘
       │                  │
       ▼                  ▼
┌──────────────┐  ┌───────────────────────┐
│  Chat Agent  │  │      Ops Agent        │
│  (LangGraph  │  │   (LangGraph 状态机)  │
│   ReAct)     │  │                       │
│              │  │  router → log_analyst │
│  Tools:      │  │    → metrics_agent    │
│  · rag_tool  │  │    → rag_recall       │
│  · log_tool  │  │    → diagnosis_agent  │
│  · time_tool │  │    → action_router    │
│  · prometheus│  │    → [HITL interrupt] │
└──────┬───────┘  │    → report_node      │
       │          └──────────┬────────────┘
       ▼                     ▼
┌─────────────────────────────────────────┐
│         RAG 混合检索层                   │
│  FAISS (向量) + BM25 (关键词) —— RRF    │
│  文档来源：data/告警处理手册.md           │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         LLM 适配层 (ai_engine)           │
│  OpenAI API ←→ 混元 API (自动切换)       │
└─────────────────────────────────────────┘
```

### HITL 审批流程

```
用户输入告警
    │
    ▼
router_agent  ──→  log_analyst ──→  metrics_agent ──→  rag_recall
    │
    ▼
diagnosis_agent（生成 RCA 报告）
    │
    ▼
action_router（检测高危关键词）
    │
    ├── 低风险 ──→  report_node  ──→ 返回最终报告
    │
    └── 高危 ──→  [LangGraph interrupt]
                      │
                      ▼
               前端弹出 HITL 审批框
                      │
               ┌──────┴──────┐
               │批准          │拒绝
               ▼             ▼
          执行建议操作    report_node（标注已拒绝）
```

---

## 项目结构

```
Agent_Rag_Ops/
├── .env                    # 环境变量（本地，不入库）
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
├── config/
│   └── config.yaml         # LLM / 检索 / 分块 参数配置
│
├── app/                    # 核心业务逻辑
│   ├── server.py           # FastAPI 主入口（所有 API 路由）
│   ├── config.py           # Settings（Pydantic + .env）
│   ├── llm.py              # LLM 客户端工厂（OpenAI / 混元）
│   │
│   ├── agents/             # LangGraph Agent
│   │   ├── state.py        # ChatState / OpsState 类型定义
│   │   ├── chat_workflow.py # Chat Agent（ReAct + SSE 生成器）
│   │   ├── ops_workflow.py # Ops Agent 多节点状态机（HITL）
│   │   └── tools/          # Agent 工具函数
│   │       ├── rag_tool.py     # RAG 知识库查询
│   │       ├── log_tool.py     # 日志拉取（Mock）
│   │       ├── prometheus.py   # Prometheus 指标（Mock）
│   │       └── time_tool.py    # 当前时间
│   │
│   ├── rag/                # RAG 检索模块
│   │   ├── indexer.py          # 文档加载 + 切分 + 建索引
│   │   ├── vector_store.py     # FAISS 向量检索
│   │   ├── bm25_retriever.py   # BM25 关键词检索
│   │   └── hybrid_retriever.py # RRF 融合检索器
│   │
│   └── observability/      # 可观测性
│       └── langfuse_callback.py # LangFuse 追踪（可选）
│
├── ai_engine/              # LLM 适配层
│   └── llm/
│       └── llm_adapter.py  # OpenAI / 混元 API 统一接口
│
├── frontend/               # Web 前端
│   ├── index.html          # 主页面
│   ├── app.js              # HexaOpsApp（对话 + Ops + HITL）
│   └── styles.css          # 样式
│
├── data/
│   ├── 告警处理手册.md      # 知识库文档源
│   └── indexes/            # 构建后的索引目录
│       ├── faiss_index/    # FAISS 向量索引
│       └── bm25_index.pkl  # BM25 索引
│
├── notebooks/
│   └── rag_evaluation.ipynb # RAG 评估（Precision / Recall / MRR）
│
└── scripts/
    └── test_chat.py        # API 单元测试脚本
```

---

## 核心模块说明

### 1. Chat Agent（`app/agents/chat_workflow.py`）

基于 LangGraph 的 ReAct（Reason + Act）框架，支持：
- **多轮对话**：`session_id` 维护会话记忆
- **工具调用**：每轮推理可调用 `rag_tool`、`log_tool`、`prometheus`、`time_tool`
- **流式输出**：`astream_events` 捕获 `on_chat_model_stream` 事件，逐 token 通过 SSE 推送

### 2. Ops Agent（`app/agents/ops_workflow.py`）

多节点 LangGraph 状态机，处理告警诊断全流程：

| 节点 | 职责 |
|------|------|
| `router_agent` | 解析告警信息，路由到各分析子任务 |
| `log_analyst` | 调用 log_tool 分析相关日志 |
| `metrics_agent` | 调用 prometheus 工具采集指标 |
| `rag_recall` | 在知识库中检索相关解决方案 |
| `diagnosis_agent` | 综合以上信息生成 RCA 根因报告 |
| `action_router` | 分析操作风险等级（low/high），决定是否触发 HITL |
| `human_approval` | HITL 节点，`interrupt_before` 暂停，等待前端审批 |
| `report_node` | 生成最终报告，附带审批状态标记 |

**高危触发关键词**：`重启 / restart / 回滚 / rollback / 删除 / delete / 清空 / truncate / 扩容 / scale / DDL / drop`

### 3. RAG 混合检索（`app/rag/`）

```
查询 query
    ├── FAISS 向量检索 → 按语义相似度排序，取 top-N
    ├── BM25 关键词检索 → 按词频 TF-IDF 排序，取 top-M
    └── RRF 融合 → score = Σ 1/(k + rank_i) → 重排后取 top-K
```

- 索引构建：调用 `POST /knowledge/index` 触发，自动扫描 `data/` 下所有 Markdown 文件
- 支持增量更新（重新调用接口即重建）

### 4. LLM 适配层（`ai_engine/llm/llm_adapter.py`）

通过 `LLM_PROVIDER` 环境变量自动切换：
- **`openai`**：调用标准 OpenAI SDK（可对接 GPT-4o / GPT-4o-mini）
- **`hunyuan`**：通过混元 OpenAI 兼容接口调用（`https://api.hunyuan.cloud.tencent.com/v1`）

---

## 快速启动

### 前置条件

- Python 3.10+（推荐 3.12）
- conda 或 venv

### 1. 克隆并创建环境

```bash
git clone <repo-url>
cd Agent_Rag_Ops

# 推荐使用 conda
conda create -n hexa-ops python=3.12 -y
conda activate hexa-ops

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写以下内容：

```dotenv
# 选择 LLM 提供商
LLM_PROVIDER=hunyuan          # 或 openai

# 混元用户（LLM_PROVIDER=hunyuan）
HUNYUAN_API_KEY=your-hunyuan-key

# OpenAI 用户（LLM_PROVIDER=openai）
OPENAI_API_KEY=your-openai-key

# 知识库索引路径（默认即可）
FAISS_INDEX_PATH=./data/indexes/faiss_index
```

### 3. 构建知识库索引

```bash
# 方式一：通过 API 触发（后端启动后调用）
curl -X POST http://localhost:8001/knowledge/index

# 方式二：直接在 Python 中运行
conda activate hexa-ops
python -c "
import sys; sys.path.insert(0, '.')
from app.rag.indexer import build_index
build_index()
print('索引构建完成')
"
```

> 知识库文档位于 `data/` 目录，支持 `.md` 格式，可自行添加告警手册、运维 SOP 等文档。

### 4. 启动后端服务

```bash
conda activate hexa-ops
cd Agent_Rag_Ops
uvicorn app.server:app --host 0.0.0.0 --port 8001 --reload
```

验证后端：
```bash
curl http://localhost:8001/health
# 响应：{"status":"ok","llm_provider":"hunyuan","llm_model":"hunyuan-turbo",...}
```

### 5. 启动前端

```bash
cd frontend
python3 -m http.server 8080
```

访问：[http://localhost:8080](http://localhost:8080)

---

## 环境变量配置

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `LLM_PROVIDER` | ✅ | LLM 提供商：`openai` / `hunyuan` | `hunyuan` |
| `HUNYUAN_API_KEY` | 条件必填 | 混元 API Key | — |
| `OPENAI_API_KEY` | 条件必填 | OpenAI API Key | — |
| `OPENAI_API_BASE` | ❌ | 自定义 OpenAI 端点 | `https://api.openai.com/v1` |
| `LLM_MODEL` | ❌ | 对话模型名 | `hunyuan-turbo` |
| `EMBEDDING_MODEL` | ❌ | Embedding 模型 | `hunyuan-embedding` |
| `EMBEDDING_DIMENSION` | ❌ | 向量维度 | `1024` |
| `FAISS_INDEX_PATH` | ❌ | FAISS 索引存储路径 | `./data/indexes/faiss_index` |
| `TOP_K` | ❌ | 检索 Top-K | `5` |
| `LANGFUSE_PUBLIC_KEY` | ❌ | LangFuse 公钥（可选追踪）| — |
| `LANGFUSE_SECRET_KEY` | ❌ | LangFuse 密钥 | — |
| `LANGFUSE_HOST` | ❌ | LangFuse 地址 | `https://cloud.langfuse.com` |

---

## API 接口文档

后端 FastAPI 自带 Swagger：打开 [http://localhost:8001/docs](http://localhost:8001/docs)

### `GET /health` — 健康检查

```json
{
  "status": "ok",
  "llm_provider": "hunyuan",
  "llm_model": "hunyuan-turbo"
}
```

---

### `POST /knowledge/index` — 构建知识库索引

触发扫描 `data/` 目录，（重）建 FAISS + BM25 索引。

**响应示例：**
```json
{
  "message": "索引构建完成",
  "doc_count": 42,
  "index_path": "./data/indexes/faiss_index"
}
```

---

### `POST /chat` — 普通对话（非流式）

**请求：**
```json
{
  "message": "Pod OOMKilled 如何处理？",
  "session_id": "session-abc123"
}
```

**响应：**
```json
{
  "session_id": "session-abc123",
  "answer": "OOMKilled 表示 Pod 因超出内存限制被强制终止..."
}
```

---

### `POST /chat/stream` — 流式对话（SSE）

**请求体同 `/chat`**，响应为 SSE 流：

```
data: {"type": "token", "content": "OOMKilled"}
data: {"type": "token", "content": " 是指..."}
data: {"type": "done", "session_id": "session-abc123"}
```

| 事件 `type` | 含义 |
|-------------|------|
| `token` | 新增文本 token |
| `done` | 流结束，附带 session_id |
| `error` | 发生错误 |

---

### `POST /ops/diagnose` — AI Ops 告警诊断

**请求：**
```json
{
  "alert_input": "生产环境 Pod payment-service-7f9d OOMKilled，内存使用 2.1GB > 限制 2GB，发生 3 次",
  "session_id": "ops-session-001"
}
```

**响应（低风险，直接完成）：**
```json
{
  "status": "done",
  "thread_id": "thread-xyz",
  "diagnosis_report": "## RCA 根因分析报告\n\n**告警描述**: ...",
  "risk_level": "low"
}
```

**响应（高风险，触发 HITL）：**
```json
{
  "status": "interrupted",
  "thread_id": "thread-xyz",
  "diagnosis_report": "## RCA 根因分析报告\n\n**建议操作**: 重启 Pod...",
  "risk_level": "high",
  "message": "高危操作需要人工审批"
}
```

> 当 `status=interrupted` 时，前端弹出 HITL 审批框，保存 `thread_id` 用于后续 approve 调用。

---

### `POST /ops/approve` — HITL 人工审批

**请求：**
```json
{
  "thread_id": "thread-xyz",
  "approved": true
}
```

**响应：**
```json
{
  "status": "done",
  "diagnosis_report": "## 最终报告\n\n已获批准并执行：重启 Pod payment-service...",
  "approved": true
}
```

---

## 运行流程详解

### 对话流程（Chat Agent）

```
用户输入
    │
    ▼
POST /chat/stream（附 session_id）
    │
    ▼
LangGraph ReAct 推理循环
    ├── [Think] LLM 决策是否调用工具
    ├── [Act]   调用工具（rag_tool / log_tool / prometheus / time_tool）
    ├── [Observe] 获取工具结果，继续推理
    └── [Final] 无需工具时，直接生成回答
    │
    ▼
SSE 流：逐 token 推送 → 前端实时渲染
```

### Ops 诊断流程

```
用户输入告警描述（AI Ops 诊断入口）
    │
    ▼
POST /ops/diagnose
    │
    ▼
LangGraph 状态机（并行 + 顺序节点）
    ├── router_agent     解析告警，构建任务上下文
    ├── log_analyst      分析日志（并行可扩展）
    ├── metrics_agent    采集 Prometheus 指标
    ├── rag_recall       知识库检索相关解决方案
    ├── diagnosis_agent  LLM 综合分析 → 生成 RCA 报告
    └── action_router    判断风险等级
            │
            ├── low  →  report_node → 返回完整报告
            │
            └── high →  [interrupt]
                             │
                        前端弹出 HITL 确认框
                             │
                      POST /ops/approve（thread_id + approved）
                             │
                        LangGraph resume → report_node
                             │
                        返回最终报告（含审批状态）
```

### 知识库构建流程

```
data/*.md 文档
    │
    ▼
Markdown 加载（langchain TextLoader）
    │
    ▼
文档切分（chunk_size=600, overlap=100）
    │
    ├──→ Embedding 模型（混元/OpenAI）→ FAISS 向量索引
    │
    └──→ BM25 分词 + 索引 → bm25_index.pkl
```

---

## RAG 评估

评估 Notebook 位于 `notebooks/rag_evaluation.ipynb`，提供三路对比：

```bash
conda activate hexa-ops
jupyter notebook notebooks/rag_evaluation.ipynb
```

**评估指标：**

| 指标 | 说明 |
|------|------|
| **Precision@K** | Top-K 结果中相关文档的比例（越高越好）|
| **Recall@K** | 相关文档被召回的比例（越高越好）|
| **MRR** | 第一个相关结果排名的倒数均值（越高越好）|

**三组对比方案：**

| 方案 | 检索策略 |
|------|---------|
| Hybrid (RRF) | FAISS 向量 + BM25 关键词，倒数秩融合 |
| FAISS Only | 纯语义向量检索 |
| BM25 Only | 纯关键词 TF-IDF 检索 |

> 若索引未构建，Notebook 自动使用 Mock 模式演示完整评估流程。

---

## 观测与追踪

系统支持接入 **LangFuse** 进行全链路追踪：

1. 在 [LangFuse](https://cloud.langfuse.com) 注册获取 API Key
2. 在 `.env` 中填写：
   ```dotenv
   LANGFUSE_PUBLIC_KEY=pk-xxx
   LANGFUSE_SECRET_KEY=sk-xxx
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
3. 后端自动注入回调，每次对话/诊断的完整 Token 用量、延迟、工具调用链路均可在 LangFuse 面板查看。

> 未配置 LangFuse 时系统静默运行，不影响任何功能。

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| Agent 框架 | LangGraph 0.2.45 |
| LLM 接入 | OpenAI SDK（兼容混元 API）|
| 向量数据库 | FAISS (CPU) |
| 关键词检索 | BM25 (rank-bm25) |
| 前端 | 原生 HTML/CSS/JS（无框架）|
| 观测 | LangFuse（可选）|
| 评估 | Jupyter Notebook + matplotlib |
| 包管理 | conda / pip |

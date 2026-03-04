"""
FastAPI 服务入口

对应 Go 项目：main.go + internal/controller/chat/

路由：
  GET  /health               健康检查
  POST /knowledge/index      构建/重建 FAISS + BM25 索引
  POST /knowledge/upload     上传知识库文档（.md/.txt）
  GET  /knowledge/inspect    查看向量库内容（调试）
  GET  /knowledge/search     测试 RAG 检索效果（调试）
  POST /chat/stream          对话（SSE 流式推送，接入 LangGraph 对话 Agent）
  POST /chat                 对话（非流式，便于调试）
  POST /ops/diagnose         运维诊断（LangGraph 多 Agent RCA，含 HITL）
  POST /ops/approve          HITL 人工审批（resume 暂停的 ops graph）
"""
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.rag.indexer import rebuild_index
from app.rag.vector_store import build_vector_store


# --------------------------------------------------------------------------
# 生命周期
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时预加载对话 Agent 图，避免第一次请求时延迟初始化过长。"""
    from app.agents.chat_workflow import get_chat_app
    get_chat_app()
    yield


# --------------------------------------------------------------------------
# App & 中间件
# --------------------------------------------------------------------------

app = FastAPI(title="Hexa-Ops AI Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# 请求/响应模型
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # 不传则每次创建新会话


class ChatResponse(BaseModel):
    session_id: str
    answer: str


# --------------------------------------------------------------------------
# 依赖
# --------------------------------------------------------------------------

def get_app_settings() -> Settings:
    return get_settings()


# --------------------------------------------------------------------------
# 接口
# --------------------------------------------------------------------------

@app.get("/health")
async def health(settings: Settings = Depends(get_app_settings)) -> JSONResponse:
    """健康检查，返回当前 LLM 配置信息。"""
    return JSONResponse(
        {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        }
    )


@app.post("/knowledge/index")
async def knowledge_index(
    settings: Settings = Depends(get_app_settings),
) -> JSONResponse:
    """
    构建/重建知识库索引（FAISS 向量索引 + BM25 倒排索引）。
    从 data/ 目录加载所有 Markdown 文档并切分写入。
    """
    # 清除 Settings 缓存，确保重建时读取最新的 .env 配置
    get_settings.cache_clear()
    try:
        stats = rebuild_index()
        return JSONResponse({"message": "ok", **stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}
_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.post("/knowledge/upload")
async def knowledge_upload(
    files: List[UploadFile] = File(...),
) -> JSONResponse:
    """
    上传知识库文档到 data/ 目录。
    支持文件类型：.md / .txt（PDF 需额外安装解析库）。
    上传后需再次调用 /knowledge/index 来重建索引。
    """
    settings = get_settings()
    # data_root 与 indexer.py 中保持一致：项目根下的 data/
    data_root = settings.faiss_index_path.parent.parent  # indexes/ -> data/

    saved: List[str] = []
    skipped: List[str] = []

    for upload in files:
        filename = upload.filename or "unnamed"
        suffix = Path(filename).suffix.lower()

        if suffix not in _ALLOWED_SUFFIXES:
            skipped.append(f"{filename} (不支持的文件类型，仅支持 .md / .txt)")
            continue

        # 读取文件内容，核查大小
        content = await upload.read()
        if len(content) > _MAX_FILE_SIZE:
            skipped.append(f"{filename} (文件过大，限 20MB)")
            continue

        dest = data_root / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        saved.append(filename)

    if not saved and skipped:
        raise HTTPException(
            status_code=400,
            detail={"message": "所有文件均被跳过", "skipped": skipped},
        )

    return JSONResponse({
        "message": f"上传完成，共 {len(saved)} 个文件。请调用 /knowledge/index 重建索引。",
        "saved": saved,
        "skipped": skipped,
        "data_dir": str(data_root),
    })


@app.get("/knowledge/inspect")
async def knowledge_inspect(limit: int = 20) -> JSONResponse:
    """
    检查向量数据库内容（调试用）。
    返回索引中存储的文档块总数及前 limit 条内容。
    """
    vs = build_vector_store()
    vs._ensure_loaded()
    if vs._faiss is None:
        return JSONResponse({"total": 0, "chunks": [], "message": "索引尚未建立，请先调用 /knowledge/index"})

    docstore: dict = vs._faiss.docstore._dict
    total = len(docstore)
    chunks = [
        {
            "id": k,
            "content": v.page_content[:400],
            "metadata": v.metadata,
        }
        for k, v in list(docstore.items())[:limit]
    ]
    return JSONResponse({"total": total, "limit": limit, "chunks": chunks})


@app.get("/knowledge/search")
async def knowledge_search(q: str, k: int = 5) -> JSONResponse:
    """
    测试 RAG 检索效果（调试用）。
    对给定查询 q 做向量相似度检索，返回 top-k 文档块。
    """
    vs = build_vector_store()
    docs = vs.similarity_search(q, k=k)
    return JSONResponse({
        "query": q,
        "k": k,
        "results": [
            {"rank": i + 1, "content": d.page_content, "metadata": d.metadata}
            for i, d in enumerate(docs)
        ],
    })


@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """
    非流式对话接口（调试用）。
    完整等待 LangGraph 跑完后返回最终答案。
    """
    from app.agents.chat_workflow import get_chat_app
    from app.observability.langfuse_callback import build_config

    session_id = req.session_id or str(uuid.uuid4())
    chat_app = get_chat_app()

    config = build_config(thread_id=session_id, session_id=session_id)
    state_input = {"messages": [HumanMessage(content=req.message)]}

    result = await chat_app.ainvoke(state_input, config=config)
    return ChatResponse(session_id=session_id, answer=result.get("answer", ""))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    流式对话接口（SSE）。

    对应 Go：internal/controller/chat/chat_v1_chat_stream.go + logic/sse/sse.go
    事件格式（与 SuperBizAgentFrontend 兼容）：
      data: {"type": "token",  "content": "..."}
      data: {"type": "done",   "session_id": "...", "answer": "..."}
      data: {"type": "error",  "message": "..."}
    """
    from app.agents.chat_workflow import get_chat_app
    from app.observability.langfuse_callback import build_config

    session_id = req.session_id or str(uuid.uuid4())
    chat_app = get_chat_app()

    async def event_generator():  # type: ignore[return]
        config = build_config(thread_id=session_id, session_id=session_id)
        state_input = {"messages": [HumanMessage(content=req.message)]}
        full_answer = ""

        try:
            async for event in chat_app.astream_events(
                state_input, config=config, version="v2"
            ):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        full_answer += token
                        payload = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"

            done_payload = json.dumps(
                {"type": "done", "session_id": session_id, "answer": full_answer},
                ensure_ascii=False,
            )
            yield f"data: {done_payload}\n\n"

        except Exception as e:
            err_payload = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class DiagnoseRequest(BaseModel):
    alert_input: str
    session_id: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool


@app.post("/ops/diagnose")
async def ops_diagnose(req: DiagnoseRequest) -> JSONResponse:
    """
    运维诊断接口（非流式，含 HITL 支持）。

    对应 Go：internal/controller/chat/chat_v1_ai_ops.go
    流程：Router Agent → [Log / Metrics / RAG] → Diagnosis Agent → Action Router
          → (high risk) human_approval(HITL 暂停) → report
          → (low  risk) report

    HITL 响应：{"status": "interrupted", "thread_id": "...", "risk_level": "high", ...}
    正常响应：{"status": "done", "thread_id": "...", "diagnosis_report": "...", ...}
    """
    from app.agents.ops_workflow import get_ops_app
    from app.agents.state import OpsState
    from app.observability.langfuse_callback import build_config

    thread_id = req.session_id or str(uuid.uuid4())
    ops_app = get_ops_app(use_hitl=True)

    init_state: OpsState = {
        "alert_input": req.alert_input,
        "alerts": [],
        "log_summary": "",
        "metrics_summary": "",
        "rag_context": [],
        "diagnosis_report": "",
        "next_action": "",
        "risk_level": "",
        "human_approved": None,
        "iteration": 0,
    }
    config = build_config(thread_id=thread_id, session_id=thread_id)
    try:
        result = await ops_app.ainvoke(init_state, config=config)

        # 检查图是否因 HITL 而暂停（interrupt_before=["human_approval"]）
        graph_state = ops_app.get_state(config)
        if graph_state.next and "human_approval" in graph_state.next:
            return JSONResponse({
                "status": "interrupted",
                "thread_id": thread_id,
                "risk_level": result.get("risk_level", "high"),
                "diagnosis_report": result.get("diagnosis_report", ""),
                "message": "诊断发现高危操作，请人工审批后继续。通过 POST /ops/approve 提交决策。",
            })

        return JSONResponse({
            "status": "done",
            "thread_id": thread_id,
            "diagnosis_report": result.get("diagnosis_report", ""),
            "log_summary": result.get("log_summary", ""),
            "metrics_summary": result.get("metrics_summary", ""),
            "risk_level": result.get("risk_level", "low"),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ops/approve")
async def ops_approve(req: ApproveRequest) -> JSONResponse:
    """
    HITL 人工审批接口。

    前端在收到 /ops/diagnose 返回 status=interrupted 后，弹出确认对话框，
    用户决策后调用本接口 resume LangGraph 图的执行。

    Body: {"thread_id": "...", "approved": true/false}
    """
    from app.agents.ops_workflow import get_ops_app
    from app.observability.langfuse_callback import build_config

    ops_app = get_ops_app(use_hitl=True)
    config = build_config(thread_id=req.thread_id, session_id=req.thread_id)

    # 验证图状态
    try:
        graph_state = ops_app.get_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"未找到会话 {req.thread_id}: {e}")

    if not graph_state.next or "human_approval" not in graph_state.next:
        raise HTTPException(status_code=400, detail="该会话未处于等待审批状态。")

    # 注入审批结果到 Checkpoint State
    ops_app.update_state(
        config,
        {"human_approved": req.approved},
        as_node="action_router",   # 以 action_router 身份更新（确保 LangGraph 接受）
    )

    # Resume 执行（传 None 表示从 Checkpoint 继续）
    try:
        result = await ops_app.ainvoke(None, config=config)
        return JSONResponse({
            "status": "done",
            "thread_id": req.thread_id,
            "approved": req.approved,
            "diagnosis_report": result.get("diagnosis_report", ""),
            "risk_level": result.get("risk_level", ""),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



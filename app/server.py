"""
FastAPI 服务入口

对应 Go 项目：main.go + internal/controller/chat/

路由：
  GET  /health               健康检查
  POST /knowledge/index      构建/重建 FAISS + BM25 索引
  POST /chat/stream          对话（SSE 流式推送，接入 LangGraph 对话 Agent）
  POST /chat                 对话（非流式，便于调试）
"""
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.rag.indexer import rebuild_index


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
    try:
        stats = rebuild_index()
        return JSONResponse({"message": "ok", **stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """
    非流式对话接口（调试用）。
    完整等待 LangGraph 跑完后返回最终答案。
    """
    from app.agents.chat_workflow import get_chat_app

    session_id = req.session_id or str(uuid.uuid4())
    chat_app = get_chat_app()

    config = {"configurable": {"thread_id": session_id}}
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

    session_id = req.session_id or str(uuid.uuid4())
    chat_app = get_chat_app()

    async def event_generator():  # type: ignore[return]
        config = {"configurable": {"thread_id": session_id}}
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


@app.post("/ops/diagnose")
async def ops_diagnose(req: DiagnoseRequest) -> JSONResponse:
    """
    运维诊断接口（非流式）。

    对应 Go：internal/controller/chat/chat_v1_ai_ops.go
    流程：Router Agent → [Log / Metrics / RAG] → Diagnosis Agent → Report
    """
    from app.agents.ops_workflow import get_ops_app
    from app.agents.state import OpsState

    ops_app = get_ops_app()
    init_state: OpsState = {
        "alert_input": req.alert_input,
        "alerts": [],
        "log_summary": "",
        "metrics_summary": "",
        "rag_context": [],
        "diagnosis_report": "",
        "next_action": "",
        "human_approved": None,
        "iteration": 0,
    }
    try:
        result = await ops_app.ainvoke(init_state)
        return JSONResponse({
            "session_id": req.session_id or "",
            "diagnosis_report": result.get("diagnosis_report", ""),
            "log_summary": result.get("log_summary", ""),
            "metrics_summary": result.get("metrics_summary", ""),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



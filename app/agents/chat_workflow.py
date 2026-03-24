"""
对话 Agent —— LangGraph StateGraph 实现

架构：
    START
      │
    rag_retrieve  ← 混合检索（BM25 + FAISS + cosine reranker）
      │
    react_agent   ← ReAct Agent（工具：query_internal_docs, get_current_time, query_pod_logs）
      │
     END

特性：
- 会话历史通过 LangGraph messages reducer 自动管理
- RAG 文档注入 system prompt（对应 Go chat_pipeline/prompt.go）
- 工具调用循环：react_agent 可多次调用工具直到 AI 不再输出 tool_calls
- 支持 LangGraph MemorySaver checkpoint 实现跨轮对话记忆
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.state import ChatState
from app.agents.tools import get_current_time, query_internal_docs, query_pod_logs
from app.llm import build_langchain_llm
from app.rag.hybrid_retriever import HybridRetriever

from langchain_core.runnables import RunnableConfig

# --------------------------------------------------------------------------
# 工具注册
# --------------------------------------------------------------------------
TOOLS = [query_internal_docs, get_current_time, query_pod_logs]

# 工具名 -> callable 映射，用于执行工具调用
TOOL_MAP = {t.name: t for t in TOOLS}

# --------------------------------------------------------------------------
# 节点：RAG 检索
# --------------------------------------------------------------------------

def rag_retrieve_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """
    从 messages 中提取最新用户问题，进行混合检索。
    检索结果存入 state['rag_docs']，供后续节点注入 prompt。

    对应 Go：chat_pipeline/retriever.go（MilvusRetriever）
    """
    # 找最后一条 HumanMessage 作为检索 query
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not query:
        return {"rag_docs": []}

    try:
        retriever = HybridRetriever.from_saved_indexes()
        docs: List[Document] = retriever.retrieve(query)
    except Exception:
        # 索引尚未构建时优雅降级
        docs = []

    return {"rag_docs": docs}


# --------------------------------------------------------------------------
# 节点：ReAct Agent（支持工具调用循环）
# --------------------------------------------------------------------------

def react_agent_node(state: ChatState, config: RunnableConfig) -> Dict[str, Any]:
    """
    ReAct Agent 节点：
    1. 将 RAG 文档拼接到 system prompt
    2. 通过 LangChain ChatOpenAI（bind_tools）调用 LLM
       — 使用 LangChain ChatModel 而非裸 SDK，确保 LangGraph astream_events
         能捕获 on_chat_model_stream 事件，SSE 流式输出才能正常工作。
    3. 若 LLM 请求工具调用，执行工具并追加结果，循环直到 AI 纯文本回复
    4. 返回最终 answer 与完整 messages 追加

    对应 Go：chat_pipeline/tools_node.go + lambda_func.go
    """
    # 使用 LangChain ChatModel（会触发 on_chat_model_stream 事件）
    llm = build_langchain_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # 1. 构建 system prompt（RAG 文档注入）
    rag_docs: List[Document] = state.get("rag_docs", [])
    rag_context = ""
    if rag_docs:
        rag_context = "\n\n".join(f"[参考文档 {i+1}]\n{d.page_content}" for i, d in enumerate(rag_docs))

    system_content = (
        "你是一位专业的 AIOps 运维专家，拥有丰富的云原生系统排查经验。\n"
        "在回答问题时，请结合提供的运维知识库文档进行分析，给出清晰、准确的回答。\n"
        "记住要善于利用工具，如果你需要查询内部文档、获取当前时间或查询日志，请调用相应的工具。\n"
        "- 查询内部文档：调用 query_internal_docs，参数为你的查询内容\n"
        "- 获取当前时间：调用 get_current_time，无需参数\n"
        "- 查询日志：调用 query_pod_logs，参数为 pod 名称和查询关键词\n\n"
        "以下是一些示例对话，展示了如何使用工具：\n"
        "用户1：请帮我查询内部文档，关键词是“故障排查”。\n"
        "Agent1：调用 query_internal_docs，参数为“故障排查”。\n"
        "工具返回1：这是关于故障排查的文档内容。\n"
        "Agent1：根据查询到的文档内容，以下是故障排查的步骤：...\n\n"
        "用户2：请告诉我当前时间。\n"
        "Agent2：调用 get_current_time，无需参数。\n"
        "工具返回2：当前时间是 2026-03-19 10:00。\n"
        "Agent2：当前时间是 2026-03-19 10:00。\n\n"
        "用户3：请查询 pod 'example-pod' 的日志，关键词是 'error'。\n"
        "Agent3：调用 query_pod_logs，参数为 pod 名称 'example-pod' 和查询关键词 'error'。\n"
        "工具返回3：以下是日志内容：...\n"
        "Agent3：根据日志内容，发现了以下问题：..."
    )
    if rag_context:
        system_content += f"\n\n【运维知识库参考文档】\n{rag_context}"

    # 2. 构建 LangChain 消息列表（system + 对话历史）
    lc_messages: List[Any] = [SystemMessage(content=system_content)] + list(state["messages"])

    # 3. ReAct 循环（最多 5 轮工具调用防止死循环）
    new_messages: List[Any] = []
    final_answer = ""

    for _ in range(5):
        # invoke 经过 LangChain 事件系统 → astream_events 可捕获 on_chat_model_stream，
        # 并通过传入 config 将 LangFuse Callback 贯穿下去
        response: AIMessage = llm_with_tools.invoke(lc_messages, config=config)
        lc_messages.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            # 纯文本回复，结束循环
            final_answer = response.content if isinstance(response.content, str) else ""
            break

        # 执行工具调用，结果追加为 ToolMessage
        for tc in response.tool_calls:
            tool_fn = TOOL_MAP.get(tc["name"])
            if tool_fn:
                try:
                    tool_result = str(tool_fn.invoke(tc["args"], config=config))
                except Exception as e:
                    tool_result = f"工具执行错误: {e}"
            else:
                tool_result = f"工具 {tc['name']} 不存在。"

            tm = ToolMessage(content=tool_result, tool_call_id=tc["id"])
            lc_messages.append(tm)
            new_messages.append(tm)

    return {
        "messages": new_messages,
        "answer": final_answer,
    }


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------

def build_chat_graph(use_memory: bool = True):
    """
    构建对话 Agent 的 LangGraph StateGraph。

    Args:
        use_memory: 是否启用 MemorySaver checkpoint（跨轮会话记忆）。

    Returns:
        compiled LangGraph app。
    """
    g = StateGraph(ChatState)

    g.add_node("rag_retrieve", rag_retrieve_node)
    g.add_node("react_agent", react_agent_node)

    g.add_edge(START, "rag_retrieve")
    g.add_edge("rag_retrieve", "react_agent")
    g.add_edge("react_agent", END)

    checkpointer = MemorySaver() if use_memory else None
    return g.compile(checkpointer=checkpointer)


# 模块级单例
_chat_app = None


def get_chat_app():
    """获取编译好的对话 Agent 图（单例）。"""
    global _chat_app
    if _chat_app is None:
        _chat_app = build_chat_graph(use_memory=True)
    return _chat_app

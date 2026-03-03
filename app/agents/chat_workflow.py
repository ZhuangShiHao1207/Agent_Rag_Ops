"""
对话 Agent —— LangGraph StateGraph 实现

对应 Go 项目：internal/ai/agent/chat_pipeline/
架构：
    START
      │
    rag_retrieve  ← 混合检索（BM25 + FAISS + cosine reranker）
      │
    react_agent   ← ReAct Agent（工具：query_internal_docs, get_current_time）
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
from app.agents.tools import get_current_time, query_internal_docs
from app.llm import build_llm_client
from app.rag.hybrid_retriever import HybridRetriever

# --------------------------------------------------------------------------
# 工具注册
# --------------------------------------------------------------------------
TOOLS = [query_internal_docs, get_current_time]

# 工具名 -> callable 映射，用于执行工具调用
TOOL_MAP = {t.name: t for t in TOOLS}

# --------------------------------------------------------------------------
# 节点：RAG 检索
# --------------------------------------------------------------------------

def rag_retrieve_node(state: ChatState) -> Dict[str, Any]:
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
# 工具模式描述（传给 LLM 的 function schema）
# --------------------------------------------------------------------------

def _build_tools_schema() -> List[Dict[str, Any]]:
    """将 LangChain tool 转换为 OpenAI function-calling schema。"""
    schemas = []
    for t in TOOLS:
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}},
            },
        })
    return schemas


_TOOLS_SCHEMA = _build_tools_schema()


# --------------------------------------------------------------------------
# 节点：ReAct Agent（支持工具调用循环）
# --------------------------------------------------------------------------

def react_agent_node(state: ChatState) -> Dict[str, Any]:
    """
    ReAct Agent 节点：
    1. 将 RAG 文档拼接到 system prompt
    2. 调用 LLM（带工具 schema）
    3. 若 LLM 请求工具调用，执行工具并追加结果，循环直到 AI 纯文本回复
    4. 返回最终 answer 与完整 messages 追加

    对应 Go：chat_pipeline/tools_node.go + lambda_func.go
    """
    client = build_llm_client()

    # 1. 构建 system prompt（RAG 文档注入）
    rag_docs: List[Document] = state.get("rag_docs", [])
    rag_context = ""
    if rag_docs:
        rag_context = "\n\n".join(f"[参考文档 {i+1}]\n{d.page_content}" for i, d in enumerate(rag_docs))

    system_content = (
        "你是一位专业的 AIOps 运维专家，拥有丰富的云原生系统排查经验。\n"
        "在回答问题时，请结合提供的运维知识库文档进行分析，给出清晰、准确的回答。\n"
        "如需查询最新信息或执行特定操作，可使用提供的工具。\n"
    )
    if rag_context:
        system_content += f"\n\n【运维知识库参考文档】\n{rag_context}"

    # 2. 构建消息列表（system + 对话历史）
    messages_for_llm = [{"role": "system", "content": system_content}]
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            messages_for_llm.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": str(tc["args"])},
                    }
                    for tc in msg.tool_calls
                ]
            messages_for_llm.append(entry)
        elif isinstance(msg, ToolMessage):
            messages_for_llm.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })

    # 3. ReAct 循环（最多 5 轮工具调用防止死循环）
    new_messages: List[Any] = []
    for _ in range(5):
        response = client.chat(messages_for_llm, tools=_TOOLS_SCHEMA)  # type: ignore[arg-type]

        # 判断是否有工具调用（response 是 str，需要检查原始 API response）
        # 重新用原始 client 调用以获取结构化信息
        raw_resp = client._client.chat.completions.create(
            model=client.settings.llm_model,
            messages=messages_for_llm,  # type: ignore[arg-type]
            tools=_TOOLS_SCHEMA,  # type: ignore[arg-type]
        )
        choice = raw_resp.choices[0]
        ai_msg_content = choice.message.content or ""
        tool_calls = choice.message.tool_calls or []

        if not tool_calls:
            # 纯文本回复，结束循环
            new_messages.append(AIMessage(content=ai_msg_content))
            return {
                "messages": new_messages,
                "answer": ai_msg_content,
            }

        # 有工具调用：记录 AI 消息，执行工具，继续循环
        ai_langchain_msg = AIMessage(
            content=ai_msg_content,
            tool_calls=[
                {"id": tc.id, "name": tc.function.name, "args": eval(tc.function.arguments)}  # noqa: S307
                for tc in tool_calls
            ],
        )
        new_messages.append(ai_langchain_msg)

        # 执行工具并追加 ToolMessage
        tool_entry: Dict[str, Any] = {
            "role": "assistant",
            "content": ai_msg_content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        }
        messages_for_llm.append(tool_entry)

        for tc in tool_calls:
            tool_fn = TOOL_MAP.get(tc.function.name)
            if tool_fn:
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_result = str(tool_fn.invoke(args))
            else:
                tool_result = f"Tool {tc.function.name} not found."

            new_messages.append(ToolMessage(content=tool_result, tool_call_id=tc.id))
            messages_for_llm.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # 超过最大轮次，返回最后一次 AI 回复
    return {"messages": new_messages, "answer": ai_msg_content}  # type: ignore[possibly-undefined]


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

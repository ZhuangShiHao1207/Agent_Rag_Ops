"""
端到端测试脚本

运行方式：
    conda run -n hexa-ops python scripts/test_chat.py [chat|tool|index|ops]
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from app.agents.chat_workflow import build_chat_graph
from app.agents.ops_workflow import build_ops_graph
from app.agents.state import OpsState
from app.rag.indexer import rebuild_index


def test_index():
    print("=" * 50)
    print("Test 1: Knowledge Index (FAISS + BM25)")
    print("=" * 50)
    result = rebuild_index()
    print("Result:", result)
    print()


async def test_chat_no_rag():
    print("=" * 50)
    print("Test 2: Chat Agent (no RAG)")
    print("=" * 50)
    app = build_chat_graph(use_memory=False)
    result = await app.ainvoke({
        "messages": [HumanMessage(content="你好，请用一句话介绍你自己")],
        "rag_docs": [],
        "answer": "",
    })
    print("Answer:", result.get("answer", "")[:300])
    print()


async def test_chat_with_tool():
    print("=" * 50)
    print("Test 3: Chat Agent (with get_current_time tool)")
    print("=" * 50)
    app = build_chat_graph(use_memory=False)
    result = await app.ainvoke({
        "messages": [HumanMessage(content="现在是几点？请用工具查询当前时间")],
        "rag_docs": [],
        "answer": "",
    })
    print("Answer:", result.get("answer", "")[:300])
    print()


async def test_ops():
    print("=" * 50)
    print("Test 4: Ops Agent (Multi-Agent State Machine)")
    print("=" * 50)
    app = build_ops_graph()
    init_state: OpsState = {
        "alert_input": "order-service Pod 频繁重启，CPU 使用率超过 80%，请分析根因",
        "alerts": [],
        "log_summary": "",
        "metrics_summary": "",
        "rag_context": [],
        "diagnosis_report": "",
        "next_action": "",
        "human_approved": None,
        "iteration": 0,
    }
    result = await app.ainvoke(init_state)
    print("--- Log Summary ---")
    print(result.get("log_summary", "")[:400])
    print("\n--- Metrics Summary ---")
    print(result.get("metrics_summary", "")[:400])
    print("\n--- Diagnosis Report ---")
    print(result.get("diagnosis_report", "")[:1000])
    print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if mode == "index":
        test_index()
    elif mode == "tool":
        asyncio.run(test_chat_with_tool())
    elif mode == "ops":
        asyncio.run(test_ops())
    else:
        asyncio.run(test_chat_no_rag())

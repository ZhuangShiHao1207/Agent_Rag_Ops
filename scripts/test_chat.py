"""
端到端测试脚本：对话 Agent + /knowledge/index

运行方式：
    conda run -n hexa-ops python scripts/test_chat.py
"""
import asyncio
import sys
import os

# 确保 app 包可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from app.agents.chat_workflow import build_chat_graph
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
    print("Test 2: Chat Agent (no RAG, 不依赖索引)")
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


async def test_rag_chat():
    """Test 4: 先构建索引，再用 RAG 检索增强对话"""
    print("=" * 50)
    print("Test 4: Chat Agent + RAG (先索引再检索)")
    print("=" * 50)
    from app.rag.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever.from_saved_indexes()
    docs = retriever.retrieve("CPU 使用率高怎么处理")
    print(f"  检索到 {len(docs)} 个文档块")
    if docs:
        print("  首个 chunk 预览:", docs[0].page_content[:100])

    app = build_chat_graph(use_memory=False)
    result = await app.ainvoke({
        "messages": [HumanMessage(content="CPU 使用率高怎么处理？请参考知识库给出排查步骤")],
        "rag_docs": [],
        "answer": "",
    })
    print("Answer:", result.get("answer", "")[:400])
    print()


async def test_memory():
    """Test 5: 多轮对话记忆"""
    print("=" * 50)
    print("Test 5: Multi-turn Memory (同一 session_id)")
    print("=" * 50)
    app = build_chat_graph(use_memory=True)
    config = {"configurable": {"thread_id": "test-session-001"}}

    r1 = await app.ainvoke(
        {"messages": [HumanMessage(content="我的名字是张三")], "rag_docs": [], "answer": ""},
        config=config,
    )
    print("Round 1:", r1.get("answer", "")[:150])

    r2 = await app.ainvoke(
        {"messages": [HumanMessage(content="你还记得我叫什么吗？")], "rag_docs": [], "answer": ""},
        config=config,
    )
    print("Round 2:", r2.get("answer", "")[:150])
    print()


async def test_sse_via_http():
    """Test 6: SSE 流式接口（需要服务在 8001 端口运行）"""
    print("=" * 50)
    print("Test 6: SSE Stream via HTTP (/chat/stream on :8001)")
    print("=" * 50)
    try:
        import httpx
        tokens = []
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:8001/chat/stream",
                json={"message": "用一句话说今天天气怎么样", "session_id": "sse-test-001"},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        data = json.loads(line[6:])
                        if data.get("type") == "token":
                            tokens.append(data["content"])
                            print(data["content"], end="", flush=True)
                        elif data.get("type") == "done":
                            print(f"\n[done] total chars: {len(''.join(tokens))}")
                            break
    except Exception as e:
        print(f"  跳过（服务未运行或 httpx 不可用）: {e}")
    print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if mode == "index":
        test_index()
    elif mode == "tool":
        asyncio.run(test_chat_with_tool())
    elif mode == "rag":
        asyncio.run(test_rag_chat())
    elif mode == "memory":
        asyncio.run(test_memory())
    elif mode == "sse":
        asyncio.run(test_sse_via_http())
    elif mode == "all":
        test_index()
        asyncio.run(test_chat_no_rag())
        asyncio.run(test_chat_with_tool())
        asyncio.run(test_rag_chat())
        asyncio.run(test_memory())
        asyncio.run(test_sse_via_http())
    else:
        asyncio.run(test_chat_no_rag())

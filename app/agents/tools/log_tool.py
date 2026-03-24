"""
日志查询工具（MCP 接入占位）。

对应 Go 项目：internal/ai/tools/query_log.go

Go 版通过 MCP 协议调用腾讯云 CLS 日志服务。
Python 版：
  - 当 MCP_LOG_SERVER_URL 未配置时，返回 Mock 日志数据供演示。
  - 后续通过 mcp-sdk 接入真实日志服务，保持接口签名不变。
"""
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool

# 确保在模块加载时读取 .env 文件
_env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(_env_path)

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    pass

MCP_LOG_SERVER_URL = os.getenv("MCP_LOG_SERVER_URL", "")
print(f"DEBUG: Current MCP_LOG_SERVER_URL is: '{MCP_LOG_SERVER_URL}'")

# Mock 日志数据（按服务名映射）
_MOCK_LOGS: dict[str, str] = {
    "order-service": """
[ERROR] 2026-03-03 16:58:12 order-service pod/order-service-7d9f8b-xk2p9
  java.lang.OutOfMemoryError: Java heap space
    at com.example.order.service.OrderProcessor.process(OrderProcessor.java:142)
    at com.example.order.service.OrderService.createOrder(OrderService.java:89)

[WARN]  2026-03-03 16:57:55 order-service
  Connection pool exhausted: waiting for available connection (timeout: 3000ms)
  Active connections: 100/100, Pending: 47

[ERROR] 2026-03-03 16:57:30 order-service
  Failed to connect to MySQL: Communications link failure
  Last packet sent to the server was 0 ms ago.
""",
    "api-server": """
[WARN]  2026-03-03 16:59:01 api-server
  High CPU detected: 87.3% (threshold: 80%)
  Top thread: GC thread consuming 23% CPU

[ERROR] 2026-03-03 16:58:45 api-server
  HTTP 503 upstream: order-service unavailable (circuit breaker open)
  Failed requests in last 60s: 342
""",
    "default": """
[INFO]  2026-03-03 17:00:00 system
  No significant log anomalies detected in the last 5 minutes.
""",
}


async def _do_mcp_query(service: str, minutes: int) -> str:
    """内部异步函数：实际执行 MCP 工具调用。"""
    try:
        # MCP SSE Client URL，如有需要可拼接具体路径，如 MCP_LOG_SERVER_URL + "/sse"
        sse_url = MCP_LOG_SERVER_URL
        
        async with sse_client(sse_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # --- 调试：看看 Server 到底提供了哪些工具 ---
                available_tools = await session.list_tools()
                print(f"DEBUG: Available tools: {[t.name for t in available_tools.tools]}")
                # ------------------------------------------

                # 假定目标 MCP server 支持的对应 tool 名称为 "query_log"？
                # 这里名称和参数目前暂时未确定
                result = await session.call_tool("query_log", {
                    "service": service,
                    "minutes": minutes
                })
                
                # result.content 是一个包含了各分类 content (text / image) 的 list
                if not result.content:
                    return "MCP 日志查询返回为空。"
                
                texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
                return "\n".join(texts)
    except Exception as e:
        return f"MCP 日志查询失败: {e}"

def _fetch_logs_via_mcp(service: str, minutes: int) -> str:
    """通过 MCP 协议查询日志（使用 mcp python sdk 同步包装）。"""
    return asyncio.run(_do_mcp_query(service, minutes))


@tool
def query_pod_logs(service: str, minutes: int = 10) -> str:
    """
    查询指定服务最近的错误日志。
    适用于：分析服务崩溃、连接失败、OOM 等故障日志。

    Args:
        service: 服务名称，例如 "order-service"、"api-server"、"payment-service"
        minutes: 查询最近多少分钟的日志，默认 10 分钟

    Returns:
        近期关键日志片段（ERROR / WARN 级别）。
    """
    if MCP_LOG_SERVER_URL:
        return _fetch_logs_via_mcp(service, minutes)

    # Mock 模式
    log = _MOCK_LOGS.get(service, _MOCK_LOGS["default"])
    return f"⚠️ [演示模式 - MCP 未连接，使用 Mock 数据]\n{service} 最近 {minutes} 分钟日志：\n{log}"

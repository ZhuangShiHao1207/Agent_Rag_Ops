"""
日志查询工具（MCP 接入占位）。

对应 Go 项目：internal/ai/tools/query_log.go

Go 版通过 MCP 协议调用腾讯云 CLS 日志服务。
Python 版：
  - 当 MCP_LOG_SERVER_URL 未配置时，返回 Mock 日志数据供演示。
  - 后续通过 mcp-sdk 接入真实日志服务，保持接口签名不变。
"""
import os
from langchain_core.tools import tool

MCP_LOG_SERVER_URL = os.getenv("MCP_LOG_SERVER_URL", "")

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


def _fetch_logs_via_mcp(service: str, minutes: int) -> str:
    """通过 MCP 协议查询日志（实际接入时替换此函数体）。"""
    try:
        import requests
        resp = requests.post(
            f"{MCP_LOG_SERVER_URL}/tools/query_log",
            json={"service": service, "minutes": minutes},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("content", "")
    except Exception as e:
        return f"MCP 日志查询失败: {e}"


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

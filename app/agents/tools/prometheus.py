"""
Prometheus 指标/告警查询工具。

对应 Go 项目：internal/ai/tools/query_metrics_alerts.go
通过 Prometheus HTTP API 查询当前活跃告警和指标数据。

当前实现：
- 读取 PROMETHEUS_URL 环境变量（默认 http://localhost:9090）
- 无 Prometheus 时返回 Mock 数据，便于本地演示
"""
import os
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
TIMEOUT = int(os.getenv("PROMETHEUS_TIMEOUT", "5"))

# Mock 告警数据（Prometheus 不可用时使用）
_MOCK_ALERTS = [
    {
        "name": "HighCPUUsage",
        "severity": "warning",
        "labels": {"instance": "10.0.0.1:9090", "job": "api-server"},
        "description": "CPU 使用率持续 5 分钟超过 80%，当前值：87.3%",
    },
    {
        "name": "PodCrashLooping",
        "severity": "critical",
        "labels": {"namespace": "production", "pod": "order-service-7d9f8b-xk2p9"},
        "description": "Pod order-service-7d9f8b-xk2p9 在过去 10 分钟内重启 5 次",
    },
]


def _query_prometheus_alerts() -> List[Dict[str, Any]]:
    """调用 Prometheus /api/v1/alerts 获取活跃告警。"""
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        alerts = []
        for a in data.get("data", {}).get("alerts", []):
            if a.get("state") == "firing":
                labels = a.get("labels", {})
                annotations = a.get("annotations", {})
                alerts.append({
                    "name": labels.get("alertname", "Unknown"),
                    "severity": labels.get("severity", "unknown"),
                    "labels": labels,
                    "description": annotations.get("description", annotations.get("summary", "")),
                })
        return alerts
    except Exception:
        return []


def _query_promql(expr: str) -> str:
    """执行 PromQL 即时查询，返回格式化结果。"""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": expr},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            return f"PromQL `{expr}` 无数据返回。"
        lines = []
        for r in results[:10]:  # 最多显示 10 条
            metric = r.get("metric", {})
            value = r.get("value", [None, "N/A"])[1]
            label_str = ", ".join(f'{k}="{v}"' for k, v in metric.items() if k != "__name__")
            lines.append(f"{label_str}: {value}")
        return "\n".join(lines)
    except Exception as e:
        return f"PromQL 查询失败: {e}"


@tool
def query_prometheus_alerts() -> str:
    """
    查询 Prometheus 当前所有 firing 状态的告警。
    返回告警名称、严重等级、标签和描述。

    适用于：运维诊断的第一步，获取当前告警全景。
    """
    alerts = _query_prometheus_alerts()
    if not alerts:
        # 使用 Mock 数据演示
        alerts = _MOCK_ALERTS
        prefix = "⚠️ [演示模式 - Prometheus 未连接，使用 Mock 数据]\n\n"
    else:
        prefix = ""

    lines = [f"{prefix}当前活跃告警（共 {len(alerts)} 条）：\n"]
    for i, a in enumerate(alerts, 1):
        lines.append(
            f"{i}. [{a['severity'].upper()}] {a['name']}\n"
            f"   描述：{a['description']}\n"
            f"   标签：{a['labels']}\n"
        )
    return "\n".join(lines)


@tool
def query_metrics(promql: str) -> str:
    """
    执行 PromQL 查询，获取特定指标数据。
    适用于：深入查询 CPU、内存、请求率、错误率等具体指标。

    Args:
        promql: PromQL 表达式，例如：
                  rate(http_requests_total{status=~"5.."}[5m])
                  container_memory_usage_bytes{namespace="production"}

    Returns:
        格式化的指标查询结果。
    """
    return _query_promql(promql)

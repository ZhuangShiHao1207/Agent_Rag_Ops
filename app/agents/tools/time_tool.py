"""
时间工具：get_current_time

对应 Go 项目：internal/ai/tools/get_current_time.go
"""
from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """
    返回当前的 UTC 时间和本地时间（Asia/Shanghai）。
    适用于：在诊断报告中记录时间戳、计算事件时间差。
    """
    utc_now = datetime.now(timezone.utc)
    local_now = datetime.now()
    return (
        f"UTC 时间：{utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"北京时间：{local_now.strftime('%Y-%m-%d %H:%M:%S')}"
    )

"""
LangFuse 全链路可观测性

对应项目规划 Module 5：
  Trace: diagnose-{alert_id} / chat-{session_id}
  ├── Span: router_agent
  ├── Span: log_analyst / metrics_agent / rag_recall
  └── Span: diagnosis_agent (token usage)

用法：
  将 get_langfuse_callback() 返回的 handler 传入 LangGraph config：
    config = {"callbacks": [get_langfuse_callback()], "configurable": {...}}

若 .env 中未配置 LANGFUSE_PUBLIC_KEY，get_langfuse_callback() 返回 None，
调用方需做 None 检查或用 [h for h in [handler] if h] 过滤空值。
"""
from __future__ import annotations

from typing import Optional

from app.config import get_settings


def get_langfuse_callback(
    session_id: str = "",
    user_id: str = "hexa-ops",
) -> Optional[object]:
    """
    返回 LangFuse CallbackHandler 实例（可直接传入 LangChain / LangGraph config）。
    若 LangFuse 未配置则返回 None。

    Args:
        session_id: 会话/告警 ID，用于 Trace 分组。
        user_id:    用户标识，显示在 LangFuse 看板。
    """
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    try:
        from langfuse.callback import CallbackHandler
        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host or "https://cloud.langfuse.com",
            session_id=session_id or None,
            user_id=user_id,
            # trace_name 在每次调用时可通过 metadata 覆盖
        )
        return handler
    except Exception:
        return None


def build_config(
    thread_id: str = "",
    session_id: str = "",
    extra: dict | None = None,
) -> dict:
    """
    构建 LangGraph invoke/astream_events 所需的 config 字典，自动注入 LangFuse callback。

    Args:
        thread_id:  LangGraph MemorySaver 的会话线程 ID。
        session_id: LangFuse Trace 分组 ID（可与 thread_id 相同）。
        extra:      额外的 configurable 字段。

    Returns:
        config dict，例如：
          {"configurable": {"thread_id": "..."}, "callbacks": [...]}
    """
    configurable: dict = {}
    if thread_id:
        configurable["thread_id"] = thread_id
    if extra:
        configurable.update(extra)

    config: dict = {}
    if configurable:
        config["configurable"] = configurable

    handler = get_langfuse_callback(session_id=session_id or thread_id)
    if handler:
        config["callbacks"] = [handler]

    return config

"""
Agent 共享状态定义（TypedDict）。

对应 Go 项目：
  - ChatState   <- chat_pipeline/types.go
  - OpsState    <- plan_execute_replan/types.go
"""
from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    对话 Agent 的共享状态。

    字段说明：
    - messages: 完整对话历史（含 HumanMessage / AIMessage / ToolMessage）。
                使用 LangGraph 内置的 add_messages reducer，支持自动追加。
    - rag_docs:  当前轮次 RAG 检索到的文档块，注入 system prompt。
    - answer:    最终回复文本，供 SSE 推流使用。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    rag_docs: List[Document]
    answer: str


class Alert(TypedDict):
    """告警信息结构，对应 Go query_metrics_alerts.go 的告警数据模型。"""
    name: str          # 告警名称，如 HighCPUUsage
    severity: str      # 严重等级：critical / warning / info
    labels: dict       # 标签，如 {instance: "10.0.0.1:9090", job: "api-server"}
    description: str   # 告警描述


class OpsState(TypedDict):
    """
    运维 Agent 的共享状态（多 Agent 状态机）。

    对应 Go 项目：plan_execute_replan/ 的 Planner → Executor → Replanner 循环。
    Python 版用 LangGraph 条件边替代预置模块，获得并行节点和 HITL 能力。

    字段说明：
    - alert_input:     原始告警描述（用户输入或 Prometheus 推送）
    - alerts:          解析后的告警列表（query_prometheus 输出）
    - log_summary:     Log Analyst Agent 输出的日志摘要
    - metrics_summary: Metrics Agent 输出的指标摘要
    - rag_context:     RAG Recall Agent 检索到的知识库文档
    - diagnosis_report:Diagnosis Agent 的根因分析报告
    - next_action:     路由决策：report / auto_fix / human_approval / replan
    - human_approved:  HITL 审批结果（Phase 5 实现）
    - iteration:       当前 Replan 迭代次数，防止无限循环
    """
    alert_input: str
    alerts: List[Alert]
    log_summary: str
    metrics_summary: str
    rag_context: List[Document]
    diagnosis_report: str
    next_action: str
    human_approved: Optional[bool]
    iteration: int

"""
运维 Agent —— LangGraph 多 Agent 状态机（含 HITL）

对应 Go 项目：internal/ai/agent/plan_execute_replan/

完整流转：
    START
      │
    router_agent          ← 解析告警、决定分析方向
      │
    ┌─┴──────────────┬────────────────┐
    │                │                │
  log_analyst   metrics_agent    rag_recall    ← 三路并行
    │                │                │
    └────────────────┴────────────────┘
                      │
              diagnosis_agent      ← RCA 根因分析
                      │ 条件边
             ┌────────┴────────┐
             │                 │
         replan_node      action_router   ← 风险分级路由
                               │ 条件边
                    ┌──────────┴──────────┐
                    │                     │
            human_approval          report_node   ← 低风险直接结束
            (HITL 审批节点)              │
                    │                    END
                   END      ← 高风险等待人工，resume 后继续

HITL 实现：
  - compile(interrupt_before=["human_approval"]) 让图在进入该节点前暂停
  - 前端通过 SSE 收到 interrupt 事件后弹确认框
  - 用户回传决策 → POST /ops/approve {thread_id, approved}
  - 服务端更新 state.human_approved 后 resume（再次 ainvoke 同 thread_id）
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, StateGraph

from app.agents.state import Alert, OpsState
from app.agents.tools.log_tool import query_pod_logs
from app.agents.tools.prometheus import query_metrics, query_prometheus_alerts
from app.llm import build_llm_client

MAX_ITERATIONS = 2

# 高危操作关键词（触发 HITL）
HIGH_RISK_KEYWORDS = [
    "重启", "restart", "回滚", "rollback", "删除", "delete",
    "清空", "truncate", "扩容", "scale", "DDL", "drop",
]


# --------------------------------------------------------------------------
# 内部帮助函数
# --------------------------------------------------------------------------

def _llm_call(system: str, user: str) -> str:
    """同步调用 LLM。"""
    client = build_llm_client()
    return client.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


# --------------------------------------------------------------------------
# 节点 1：Router Agent —— 解析告警，决定分析方向
# --------------------------------------------------------------------------

def router_agent_node(state: OpsState) -> Dict[str, Any]:
    """
    对应 Go：Planner 的第 1 步，解析告警并决定要调用哪些工具。
    输出：alerts 列表 + next_action 路由决策。
    """
    alert_input = state.get("alert_input", "")

    # 获取 Prometheus 当前告警
    raw_alerts = query_prometheus_alerts.invoke({})

    system = (
        "你是运维告警路由专家。根据告警信息，判断需要进行哪些分析："
        "logs（日志分析）、metrics（指标分析）、rag（知识库检索），可以多选。"
        "输出 JSON 格式：{\"services\": [\"服务名\"], \"need_logs\": bool, \"need_metrics\": bool, \"need_rag\": bool, \"summary\": \"一句话概述\"}"
    )
    user = f"用户描述：{alert_input}\n\nPrometheus 告警：\n{raw_alerts}"

    try:
        resp = _llm_call(system, user)
        # 提取 JSON
        start = resp.find("{")
        end = resp.rfind("}") + 1
        parsed = json.loads(resp[start:end]) if start != -1 else {}
    except Exception:
        parsed = {"services": ["api-server"], "need_logs": True, "need_metrics": True, "need_rag": True, "summary": alert_input}

    # 构建 alerts 列表
    alerts: List[Alert] = [
        Alert(
            name="RouterParsed",
            severity="warning",
            labels={"services": str(parsed.get("services", []))},
            description=parsed.get("summary", alert_input),
        )
    ]

    return {
        "alerts": alerts,
        "next_action": json.dumps({
            "need_logs": parsed.get("need_logs", True),
            "need_metrics": parsed.get("need_metrics", True),
            "need_rag": parsed.get("need_rag", True),
            "services": parsed.get("services", ["api-server"]),
        }),
        "risk_level": "",
        "iteration": state.get("iteration", 0),
    }


# --------------------------------------------------------------------------
# 节点 2a：Log Analyst Agent
# --------------------------------------------------------------------------

def log_analyst_node(state: OpsState) -> Dict[str, Any]:
    """
    对应 Go：Executor 调用 query_log.go（MCP）工具。
    查询相关服务的错误日志并提炼摘要。
    """
    try:
        routing = json.loads(state.get("next_action", "{}"))
        services: List[str] = routing.get("services", ["api-server"])
    except Exception:
        services = ["api-server"]

    log_parts = []
    for svc in services[:3]:  # 最多查 3 个服务
        raw = query_pod_logs.invoke({"service": svc, "minutes": 10})
        log_parts.append(f"=== {svc} ===\n{raw}")

    all_logs = "\n\n".join(log_parts)

    summary = _llm_call(
        system=(
            "你是日志分析专家。请从以下日志中提炼关键异常信息："
            "重点关注 ERROR/WARN 级别，识别根因线索（OOM、连接失败、超时等），输出 200 字以内的分析摘要。"
        ),
        user=all_logs,
    )
    return {"log_summary": summary}


# --------------------------------------------------------------------------
# 节点 2b：Metrics Agent
# --------------------------------------------------------------------------

def metrics_agent_node(state: OpsState) -> Dict[str, Any]:
    """
    对应 Go：Executor 调用 query_metrics_alerts.go 工具。
    查询 Prometheus 指标并提炼摘要。
    """
    try:
        routing = json.loads(state.get("next_action", "{}"))
        services: List[str] = routing.get("services", ["api-server"])
    except Exception:
        services = ["api-server"]

    # 查询关键指标
    cpu_data = query_metrics.invoke({"promql": f'rate(process_cpu_seconds_total[5m])'})
    alert_data = query_prometheus_alerts.invoke({})

    raw = f"CPU 指标：\n{cpu_data}\n\n活跃告警：\n{alert_data}"

    summary = _llm_call(
        system=(
            "你是指标分析专家。请从以下 Prometheus 指标和告警数据中提炼关键信息："
            "重点识别资源瓶颈（CPU/内存/连接数），输出 200 字以内的指标摘要。"
        ),
        user=raw,
    )
    return {"metrics_summary": summary}


# --------------------------------------------------------------------------
# 节点 2c：RAG Recall Agent
# --------------------------------------------------------------------------

def rag_recall_node(state: OpsState) -> Dict[str, Any]:
    """
    对应 Go：Executor 调用 query_internal_docs.go 工具。
    从知识库检索与当前告警相关的运维手册文档。
    """
    alerts = state.get("alerts", [])
    query = state.get("alert_input", "")
    if alerts:
        query = f"{query} {alerts[0].get('description', '')}"

    try:
        from app.rag.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever.from_saved_indexes()
        docs: List[Document] = retriever.retrieve(query, top_k=5)
    except Exception:
        docs = []

    return {"rag_context": docs}


# --------------------------------------------------------------------------
# 节点 3：Diagnosis Agent —— 根因分析 RCA
# --------------------------------------------------------------------------

def diagnosis_agent_node(state: OpsState) -> Dict[str, Any]:
    """
    对应 Go：Planner（使用 DeepSeek-R1 推理，Python 版用 hunyuan-turbo 或可配置）。
    综合日志、指标、知识库输出根因分析报告。
    """
    rag_context = "\n\n".join(
        f"[知识库文档 {i+1}]\n{d.page_content}"
        for i, d in enumerate(state.get("rag_context", []))
    ) or "暂无知识库文档"

    system = (
        "你是资深 SRE 运维专家，擅长云原生系统根因分析。\n"
        "请综合以下信息，给出结构化的根因分析报告（RCA Report）：\n"
        "1. 故障概述（1-2句）\n"
        "2. 根因分析（列举 1-3 个可能根因，附置信度）\n"
        "3. 影响范围\n"
        "4. 建议处置步骤（按优先级排序）\n"
        "5. 预防措施"
    )
    user = (
        f"【原始告警】\n{state.get('alert_input', '')}\n\n"
        f"【日志分析】\n{state.get('log_summary', '未获取')}\n\n"
        f"【指标分析】\n{state.get('metrics_summary', '未获取')}\n\n"
        f"【知识库参考】\n{rag_context}"
    )

    report = _llm_call(system=system, user=user)

    # 判断是否需要 replan（信息不足时）
    iteration = state.get("iteration", 0)
    if ("信息不足" in report or "无法确定" in report) and iteration < MAX_ITERATIONS:
        next_action = "replan"
    else:
        next_action = "action_router"

    return {
        "diagnosis_report": report,
        "next_action": next_action,
        "iteration": iteration + 1,
    }


# --------------------------------------------------------------------------
# 节点 4：Action Router —— 风险分级
# --------------------------------------------------------------------------

def action_router_node(state: OpsState) -> Dict[str, Any]:
    """
    判断诊断报告中的建议是否包含高危操作。
    高危 → human_approval（HITL 暂停）
    低危 → report_node（直接输出）
    """
    report = state.get("diagnosis_report", "")
    is_high_risk = any(kw in report for kw in HIGH_RISK_KEYWORDS)
    risk_level = "high" if is_high_risk else "low"
    return {
        "risk_level": risk_level,
        "next_action": "human_approval" if is_high_risk else "report",
    }


# --------------------------------------------------------------------------
# 节点 5：Human Approval（HITL）
# --------------------------------------------------------------------------

def human_approval_node(state: OpsState) -> Dict[str, Any]:
    """
    HITL 审批节点。

    当 compile(interrupt_before=["human_approval"]) 时，LangGraph 在进入
    本节点前暂停（Checkpoint 保存当前 State），前端通过 SSE 收到 interrupt 事件。
    用户通过 POST /ops/approve 提交决策后，调用方 resume（同 thread_id 再次调用）。

    本节点执行时 human_approved 已由调用方通过 update_state 注入。
    """
    approved = state.get("human_approved")
    if approved is None:
        raise NodeInterrupt("等待人工审批，请通过 /ops/approve 接口提交决策。")

    if not approved:
        return {
            "diagnosis_report": (
                state.get("diagnosis_report", "") +
                "\n\n---\n⚠️ **[人工审批]** 高危操作已被拒绝，系统保持当前状态，建议人工介入处理。"
            ),
            "next_action": "report",
        }

    return {
        "diagnosis_report": (
            state.get("diagnosis_report", "") +
            "\n\n---\n✅ **[人工审批]** 高危操作已获批准，建议按照上述步骤执行。请做好回滚预案。"
        ),
        "next_action": "report",
    }


# --------------------------------------------------------------------------
# 节点 6a：Report Node —— 输出最终报告
# --------------------------------------------------------------------------

def report_node(state: OpsState) -> Dict[str, Any]:
    """终止节点，标记 next_action = report。"""
    return {"next_action": "report"}


# --------------------------------------------------------------------------
# 节点 6b：Replan Node —— 补充信息后重新路由
# --------------------------------------------------------------------------

def replan_node(state: OpsState) -> Dict[str, Any]:
    """
    对应 Go：Replanner。
    当 Diagnosis Agent 判断信息不足时，扩大查询范围重新分析。
    """
    current_report = state.get("diagnosis_report", "")
    system = (
        "根据当前诊断报告，判断还需要收集哪些额外信息。"
        "输出 JSON：{\"services\": [\"额外服务名\"], \"need_logs\": bool, \"need_metrics\": bool, \"need_rag\": bool}"
    )
    user = f"当前报告（信息不足部分）：\n{current_report}"

    try:
        resp = _llm_call(system, user)
        start = resp.find("{")
        end = resp.rfind("}") + 1
        parsed = json.loads(resp[start:end]) if start != -1 else {}
    except Exception:
        parsed = {"services": [], "need_logs": True, "need_metrics": False, "need_rag": True}

    return {
        "next_action": json.dumps({
            "need_logs": parsed.get("need_logs", True),
            "need_metrics": parsed.get("need_metrics", False),
            "need_rag": parsed.get("need_rag", True),
            "services": parsed.get("services", []),
        }),
    }


# --------------------------------------------------------------------------
# 条件边路由函数
# --------------------------------------------------------------------------

def route_after_diagnosis(state: OpsState) -> Literal["action_router", "replan_node"]:
    """Diagnosis Agent 后的路由：action_router 或 replan。"""
    if state.get("next_action") == "replan" and state.get("iteration", 0) <= MAX_ITERATIONS:
        return "replan_node"
    return "action_router"


def route_after_action_router(state: OpsState) -> Literal["human_approval", "report_node"]:
    """Action Router 后的路由：高危 → human_approval，低危 → report_node。"""
    return "human_approval" if state.get("risk_level") == "high" else "report_node"


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------

def build_ops_graph(use_hitl: bool = True):
    """
    构建运维 Agent StateGraph。

    Args:
        use_hitl: True  → 启用 HITL（interrupt_before=["human_approval"]），需 MemorySaver。
                  False → 不中断，适合脚本测试。
    """
    g = StateGraph(OpsState)

    # 注册节点
    g.add_node("router_agent", router_agent_node)
    g.add_node("log_analyst", log_analyst_node)
    g.add_node("metrics_agent", metrics_agent_node)
    g.add_node("rag_recall", rag_recall_node)
    g.add_node("diagnosis_agent", diagnosis_agent_node)
    g.add_node("action_router", action_router_node)
    g.add_node("human_approval", human_approval_node)
    g.add_node("report_node", report_node)
    g.add_node("replan_node", replan_node)

    # 主干边
    g.add_edge(START, "router_agent")

    # router → 三路并行
    g.add_edge("router_agent", "log_analyst")
    g.add_edge("router_agent", "metrics_agent")
    g.add_edge("router_agent", "rag_recall")

    # 三路汇聚 → diagnosis
    g.add_edge("log_analyst", "diagnosis_agent")
    g.add_edge("metrics_agent", "diagnosis_agent")
    g.add_edge("rag_recall", "diagnosis_agent")

    # diagnosis → action_router | replan
    g.add_conditional_edges(
        "diagnosis_agent",
        route_after_diagnosis,
        {"action_router": "action_router", "replan_node": "replan_node"},
    )

    # replan → 重新进入三路分析
    g.add_edge("replan_node", "log_analyst")

    # action_router → human_approval | report_node
    g.add_conditional_edges(
        "action_router",
        route_after_action_router,
        {"human_approval": "human_approval", "report_node": "report_node"},
    )

    # human_approval → report_node
    g.add_edge("human_approval", "report_node")
    g.add_edge("report_node", END)

    checkpointer = MemorySaver() if use_hitl else None
    interrupt_before = ["human_approval"] if use_hitl else []
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


# 模块级单例（use_hitl=True，生产模式）
_ops_app = None


def get_ops_app(use_hitl: bool = True):
    """获取编译好的运维 Agent 图（单例）。"""
    global _ops_app
    if _ops_app is None:
        _ops_app = build_ops_graph(use_hitl=use_hitl)
    return _ops_app

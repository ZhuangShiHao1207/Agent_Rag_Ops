"""
运维 Agent —— LangGraph 多 Agent 状态机

对应 Go 项目：internal/ai/agent/plan_execute_replan/

架构（状态机流转）：
    START
      │
    router_agent          ← 解析告警，决定路由（对应 Go Planner 第 1 步）
      │ 条件边
    ┌─┴──────────────┬────────────────┐
    │                │                │
  log_analyst     metrics_agent    rag_recall    ← 并行分析（对应 Go Executor 三类工具）
    │                │                │
    └────────────────┴────────────────┘
              汇聚（Diagnosis Agent 前置等待）
                      │
              diagnosis_agent      ← 根因分析 RCA（对应 Go Planner 推理阶段）
                      │ 条件边
            ┌─────────┴──────────┐
            │                    │
        report_node          replan_node   ← 信息不足则 Replan（对应 Go Replanner）
            │
           END

特性：
- 三路分析节点共享同一 OpsState，通过字段隔离输出（无竞争条件）
- Diagnosis Agent 用强力 hunyuan-pro 模型（可配置为 deepseek-r1）
- Replan 最多 2 轮防止死循环
- Phase 5 在 report_node 前插入 HITL 审批节点
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.state import Alert, OpsState
from app.agents.tools.log_tool import query_pod_logs
from app.agents.tools.prometheus import query_metrics, query_prometheus_alerts
from app.agents.tools.rag_tool import query_internal_docs
from app.llm import build_llm_client

MAX_ITERATIONS = 2  # Replan 最大迭代次数


# --------------------------------------------------------------------------
# 内部帮助函数
# --------------------------------------------------------------------------

def _llm_call(system: str, user: str, model_override: str | None = None) -> str:
    """同步调用 LLM，支持指定不同模型（如 hunyuan-pro 用于深度推理）。"""
    client = build_llm_client()
    # 临时替换模型（运维诊断用更强的模型）
    if model_override:
        orig_model = client.settings.llm_model
        client.settings.__dict__["llm_model"] = model_override

    result = client.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    if model_override:
        client.settings.__dict__["llm_model"] = orig_model

    return result


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
        next_action = "report"

    return {
        "diagnosis_report": report,
        "next_action": next_action,
        "iteration": iteration + 1,
    }


# --------------------------------------------------------------------------
# 节点 4a：Report Node —— 输出最终报告
# --------------------------------------------------------------------------

def report_node(state: OpsState) -> Dict[str, Any]:
    """直接返回诊断报告（无需额外处理，Phase 5 在此前插入 HITL）。"""
    # LangGraph 要求节点必须返回至少一个 state 字段；
    # next_action 置为 "report" 标识最终状态，其余字段保持不变。
    return {"next_action": "report"}


# --------------------------------------------------------------------------
# 节点 4b：Replan Node —— 补充信息后重新路由
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

def route_after_diagnosis(state: OpsState) -> Literal["report_node", "replan_node"]:
    """Diagnosis Agent 后的路由：report 或 replan。"""
    if state.get("next_action") == "replan" and state.get("iteration", 0) <= MAX_ITERATIONS:
        return "replan_node"
    return "report_node"


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------

def build_ops_graph():
    """
    构建运维 Agent 的 LangGraph StateGraph。

    三路分析节点（log/metrics/rag）并发执行后汇聚到 diagnosis_agent。
    """
    g = StateGraph(OpsState)

    # 注册节点
    g.add_node("router_agent", router_agent_node)
    g.add_node("log_analyst", log_analyst_node)
    g.add_node("metrics_agent", metrics_agent_node)
    g.add_node("rag_recall", rag_recall_node)
    g.add_node("diagnosis_agent", diagnosis_agent_node)
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

    # diagnosis 条件边
    g.add_conditional_edges(
        "diagnosis_agent",
        route_after_diagnosis,
        {"report_node": "report_node", "replan_node": "replan_node"},
    )

    # replan → 重新进入三路分析
    g.add_edge("replan_node", "log_analyst")
    g.add_edge("report_node", END)

    return g.compile()


# 模块级单例
_ops_app = None


def get_ops_app():
    """获取编译好的运维 Agent 图（单例）。"""
    global _ops_app
    if _ops_app is None:
        _ops_app = build_ops_graph()
    return _ops_app

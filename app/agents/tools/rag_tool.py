"""
RAG 工具：query_internal_docs

对应 Go 项目：internal/ai/tools/query_internal_docs.go
功能：从向量知识库（FAISS + BM25 混合检索）检索运维相关文档。
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.rag.hybrid_retriever import HybridRetriever

# 模块级单例，避免每次工具调用重复加载索引
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever.from_saved_indexes()
    return _retriever


@tool
def query_internal_docs(query: str) -> str:
    """
    从运维知识库检索与查询最相关的文档片段。
    适用于：查询告警处理手册、排查步骤、历史故障案例等。

    Args:
        query: 自然语言检索查询，例如 "CPU 使用率高怎么处理"

    Returns:
        拼接的相关文档片段文本。
    """
    retriever = _get_retriever()
    docs: List[Document] = retriever.retrieve(query)
    if not docs:
        return "知识库中未找到相关内容。"
    return "\n\n---\n\n".join(d.page_content for d in docs)

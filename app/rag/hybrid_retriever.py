"""
混合检索模块：BM25 稀疏检索 + FAISS 向量检索 → RRF 融合 → 可选 Reranker 精排。

流程：
    query
    ├── BM25 关键词检索 (Top-K) ──────────────────────────────┐
    └── FAISS 向量检索  (Top-K) ──────────────────────────────┤
                                                    RRF Fusion (Top-N)
                                                              │
                                              Reranker 精排 (Top-K，可选)
                                                              │
                                                     最终 Document 列表

说明：
- RRF（Reciprocal Rank Fusion）：把两路排名融合，无需归一化分数，容错性强。
- Reranker 接口为可插拔 callable，默认使用 embedding cosine 相似度对候选集重排。
  若后续安装 sentence-transformers / FlagEmbedding，只需替换 reranker 参数即可。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document

from app.config import get_settings
from app.llm import build_llm_client
from app.rag.bm25_retriever import BM25Retriever
from app.rag.vector_store import VectorStore, build_vector_store


# -------------------------------------------------------------------
# RRF 融合
# -------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: List[List[Document]],
    k_constant: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion。

    Args:
        ranked_lists: 多路检索结果，每路已按相关性降序排列。
        k_constant: RRF 超参数，论文默认 60。

    Returns:
        按 RRF 分数降序的 (Document, score) 列表，已去重（以 page_content 为 key）。
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content  # 以内容去重
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k_constant)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [(doc_map[k], scores[k]) for k in sorted_keys]


# -------------------------------------------------------------------
# 默认 Reranker：embedding cosine 相似度
# -------------------------------------------------------------------

def _cosine_reranker(
    query: str,
    candidates: List[Document],
    top_k: int,
) -> List[Document]:
    """
    使用配置中的 embedding 模型计算 query 与候选文档的余弦相似度，重新排序。

    当 sentence-transformers / BGE 可用时，可用更强的交叉编码器替换本函数。
    """
    client = build_llm_client()
    # 批量向量化：query + 所有候选文档
    all_texts = [query] + [d.page_content for d in candidates]
    all_vecs = np.array(client.embed_documents(all_texts), dtype=np.float32)

    query_vec = all_vecs[0]
    doc_vecs = all_vecs[1:]

    # 余弦相似度
    query_norm = np.linalg.norm(query_vec) + 1e-10
    doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-10
    sims = (doc_vecs @ query_vec) / (doc_norms * query_norm)

    ranked_indices = np.argsort(sims)[::-1][:top_k]
    return [candidates[i] for i in ranked_indices]


# -------------------------------------------------------------------
# HybridRetriever 主类
# -------------------------------------------------------------------

class HybridRetriever:
    """
    混合检索器：BM25 + FAISS → RRF → Reranker。

    Args:
        vector_store: VectorStore 实例（FAISS，后续可换为 Milvus）。
        bm25: BM25Retriever 实例。
        reranker: 可选 callable(query, candidates, top_k) -> List[Document]。
                  默认使用 embedding cosine 相似度。传入 None 则跳过 reranker。
        fetch_k: RRF 之前每路各取多少候选，默认 20。
        top_k: 最终返回数量，默认读取 settings.top_k。
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25: Optional[BM25Retriever] = None,
        reranker: Optional[Callable[[str, List[Document], int], List[Document]]] = _cosine_reranker,
        fetch_k: int = 20,
        top_k: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._vs = vector_store or build_vector_store()
        self._bm25 = bm25
        self._reranker = reranker
        self._fetch_k = fetch_k
        self._top_k = top_k or settings.top_k

    # ------------------------------------------------------------------
    # 加载 BM25（从持久化文件）
    # ------------------------------------------------------------------

    @classmethod
    def from_saved_indexes(
        cls,
        chroma_persist_dir: Optional[Path] = None,
        bm25_index_path: Optional[Path] = None,
        **kwargs,
    ) -> "HybridRetriever":
        """
        从磁盘加载 FAISS 和 BM25 索引，构建 HybridRetriever。
        """
        settings = get_settings()
        faiss_path = chroma_persist_dir or settings.chroma_persist_dir
        bm25_path = bm25_index_path or (faiss_path.parent / "bm25_index.pkl")

        vs = build_vector_store(persist_dir=faiss_path)

        bm25 = None
        if bm25_path.exists():
            bm25 = BM25Retriever.load(bm25_path)

        return cls(vector_store=vs, bm25=bm25, **kwargs)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        混合检索主入口。

        Returns:
            经过 RRF 融合（+ 可选 Reranker）的 top_k 个 Document。
        """
        k = top_k or self._top_k
        fetch = max(self._fetch_k, k * 4)  # 候选集数量至少是 top_k 的 4 倍

        ranked_lists: List[List[Document]] = []

        # 1. FAISS 向量检索
        faiss_results = self._vs.similarity_search(query, k=fetch)
        if faiss_results:
            ranked_lists.append(faiss_results)

        # 2. BM25 关键词检索
        if self._bm25 is not None:
            bm25_results = self._bm25.search(query, k=fetch)
            if bm25_results:
                ranked_lists.append(bm25_results)

        if not ranked_lists:
            return []

        # 3. RRF 融合，取 top min(fetch, len) 候选
        fused = _rrf_fuse(ranked_lists)
        candidates = [doc for doc, _ in fused[: max(fetch, 20)]]

        # 4. Reranker 精排（可选）
        if self._reranker and len(candidates) > k:
            return self._reranker(query, candidates, k)

        return candidates[:k]

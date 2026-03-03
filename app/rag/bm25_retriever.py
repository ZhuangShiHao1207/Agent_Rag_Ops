"""
BM25 稀疏检索模块。

使用 rank-bm25 构建倒排索引，支持精确匹配（如错误码 ERROR-5002、OOMKilled）。
索引序列化到磁盘，与 FAISS 向量索引共用同一父目录。
"""
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """
    简单分词：按空白 + 标点切分，中文按字符切分以支持精确匹配。
    """
    import re
    # 先按非汉字切分，再将汉字序列按字切开
    tokens: List[str] = []
    for part in re.split(r"[\s\u3000\.,，。！？!?；;:：\-—/\\|]+", text.lower()):
        if not part:
            continue
        # 如果片段包含汉字，将连续汉字逐字拆开，其他字符保留
        cjk_parts = re.split(r"([\u4e00-\u9fff]+)", part)
        for seg in cjk_parts:
            if re.match(r"[\u4e00-\u9fff]+", seg):
                tokens.extend(list(seg))  # 汉字按字切
            elif seg:
                tokens.append(seg)
    return tokens


class BM25Retriever:
    """
    BM25 倒排索引检索器，封装 rank-bm25，并实现 save/load 持久化。
    接口与 VectorStore.similarity_search 对齐，便于 HybridRetriever 统一调用。
    """

    def __init__(self, bm25: BM25Okapi, docs: List[Document]) -> None:
        self._bm25 = bm25
        self._docs = docs

    # ------------------------------------------------------------------
    # 构建 & 持久化
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        docs: List[Document],
        save_path: Optional[Path] = None,
    ) -> "BM25Retriever":
        """
        从 Document 列表构建 BM25 索引，可选择持久化到磁盘。
        """
        corpus = [_tokenize(d.page_content) for d in docs]
        bm25 = BM25Okapi(corpus)
        instance = cls(bm25=bm25, docs=docs)
        if save_path is not None:
            instance.save(save_path)
        return instance

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "docs": self._docs}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(bm25=data["bm25"], docs=data["docs"])

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        返回 BM25 分数最高的 k 个文档。
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # 取 Top-k 索引（分数从高到低）
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in top_indices]

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        返回 (Document, score) 列表，供 HybridRetriever 做 RRF 融合。
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in top_indices]

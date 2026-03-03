from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from app.llm import build_llm_client


class VectorStore:
    """
    向量存储抽象层。

    当前实现基于本地 FAISS，后续可以在不改上层代码的情况下替换为 Milvus / PGVector 等远程向量库。
    """

    def __init__(self, index_path: Path | None = None) -> None:
        settings = get_settings()
        self.index_path = index_path or settings.faiss_index_path
        self._faiss: FAISS | None = None
        self._client = build_llm_client()

    def _ensure_loaded(self) -> None:
        if self._faiss is not None:
            return
        if self.index_path.exists():
            self._faiss = FAISS.load_local(
                folder_path=str(self.index_path),
                embeddings=self._client,  # LLMClient 需实现 LangChain Embeddings 接口
                allow_dangerous_deserialization=True,
            )
        else:
            # 延迟初始化为空索引，由上层通过 add_documents 构建
            self._faiss = None

    def add_documents(self, docs: Sequence[Document]) -> None:
        """
        将一批文档插入索引。如果索引不存在，则新建。
        """
        if not docs:
            return
        embeddings = self._client  # 作为 embeddings 对象传入

        if self._faiss is None:
            self._faiss = FAISS.from_documents(docs, embedding=embeddings)
        else:
            self._faiss.add_documents(docs)

    def save(self) -> None:
        """
        将当前索引持久化到磁盘。
        """
        if self._faiss is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.save_local(str(self.index_path))

    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        """
        简单向量相似度检索（后续会被混合检索封装调用）。
        """
        self._ensure_loaded()
        if self._faiss is None:
            return []

        settings = get_settings()
        k = k or settings.top_k
        return self._faiss.similarity_search(query, k=k)


def build_vector_store(index_path: Path | None = None) -> VectorStore:
    return VectorStore(index_path=index_path)


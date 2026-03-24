from pathlib import Path
from typing import List, Sequence
import hashlib
from datetime import datetime
import shutil

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.llm import build_llm_client


class VectorStore:
    """
    向量存储抽象层。
    当前实现基于本地 Chroma，支持增量覆盖和自动持久化。
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._client = build_llm_client()
        self._chroma: Chroma | None = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._chroma is not None:
            return

        self.persist_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            # 初始化加载 Chroma，不需要区分是新建还是读取，它会自动处理持久化目录
            self._chroma = Chroma(
                collection_name="knowledge_base",
                embedding_function=self._client,
                persist_directory=str(self.persist_dir),
                client_settings=ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=str(self.persist_dir),
                ),
            )
        except KeyError as e:
            # 常见于 chromadb 版本切换后，本地持久化元数据不兼容（如缺少 _type）
            if str(e) != "'_type'":
                raise
            self._recover_incompatible_persist_dir()
            self._chroma = Chroma(
                collection_name="knowledge_base",
                embedding_function=self._client,
                persist_directory=str(self.persist_dir),
                client_settings=ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=str(self.persist_dir),
                ),
            )

    def _recover_incompatible_persist_dir(self) -> None:
        if not self.persist_dir.exists():
            self._clear_chroma_system_cache()
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.persist_dir.with_name(f"{self.persist_dir.name}_backup_incompatible_{ts}")
        shutil.move(str(self.persist_dir), str(backup_dir))
        self._clear_chroma_system_cache()

    @staticmethod
    def _clear_chroma_system_cache() -> None:
        # chromadb 在进程内缓存 client/system，目录迁移后需清缓存再重建连接
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass

    def _generate_doc_id(self, doc: Document) -> str:
        """为 Document 生成唯一 ID（基于来源、分块位置和内容哈希）"""
        source = doc.metadata.get("source", "unknown")
        # 很多切分器也会带 loc 信息，没有的话用文本代替
        content = doc.page_content.encode('utf-8')
        return hashlib.md5(f"{source}_{content}".encode('utf-8')).hexdigest()

    def add_documents(self, docs: Sequence[Document]) -> None:
        """
        带 ID 插入文档。如果 ID 已存在，ChromaDB 会自动进行覆盖计算。
        """
        if not docs:
            return
        self._ensure_loaded()
        
        # 生成唯一 ID
        ids = [self._generate_doc_id(doc) for doc in docs]
        
        # add_documents 支持传入 ids 并在底层利用 Chroma 的 upsert
        if self._chroma is not None:
            self._chroma.add_documents(docs, ids=ids)

    def save(self) -> None:
        """
        旧接口兼容：Chroma 会根据配置自动持久化，不再需要手动 save
        """
        pass

    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        """
        向量相似度检索。
        """
        self._ensure_loaded()
        if self._chroma is None:
            return []

        settings = get_settings()
        k = k or settings.top_k
        return self._chroma.similarity_search(query, k=k)


def build_vector_store(persist_dir: Path | None = None) -> VectorStore:
    return VectorStore(persist_dir=persist_dir)



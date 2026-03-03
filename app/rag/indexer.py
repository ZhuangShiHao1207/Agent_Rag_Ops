from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.vector_store import build_vector_store
from app.rag.bm25_retriever import BM25Retriever


def _load_markdown_docs(root: Path) -> List[Document]:
    """
    从指定目录递归加载所有 Markdown 文档。
    显式指定 loader_cls=TextLoader，无需安装 unstructured 包。
    """
    loader = DirectoryLoader(
        str(root),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load()


def _split_docs(docs: List[Document]) -> List[Document]:
    """
    使用 Markdown 头分割，将每个文档切成较小的 chunk。
    每个原始文档单独调用 split_text(str)，避免将 generator 传入导致 bug。
    """
    if not docs:
        return []

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=False,
    )
    chunks: List[Document] = []
    for doc in docs:
        # split_text 只接受 str，逐个文档调用
        sub_docs = splitter.split_text(doc.page_content)
        # 将原始 metadata（文件路径等）合并到每个子文档
        for d in sub_docs:
            merged = dict(doc.metadata)
            merged.update(d.metadata)
            d.metadata = merged
        chunks.extend(sub_docs)
    return chunks


def rebuild_index() -> dict:
    """
    从 data/ 目录重新构建 FAISS 向量索引 + BM25 倒排索引。

    返回简单的统计信息，便于 /knowledge/index 接口展示。
    """
    settings = get_settings()
    data_root = Path(__file__).resolve().parent.parent.parent / "data"

    docs = _load_markdown_docs(data_root)
    chunks = _split_docs(docs)

    if not chunks:
        return {
            "data_root": str(data_root),
            "faiss_index_path": str(settings.faiss_index_path),
            "file_count": len(docs),
            "chunk_count": 0,
            "warning": "No chunks generated. Check data/ directory.",
        }

    # 1. 构建 FAISS 向量索引
    vs = build_vector_store()
    vs.add_documents(chunks)
    vs.save()

    # 2. 同步构建 BM25 倒排索引（持久化到与 FAISS 相同的父目录）
    bm25_path = settings.faiss_index_path.parent / "bm25_index.pkl"
    bm25 = BM25Retriever.build(chunks, save_path=bm25_path)  # noqa: F841

    return {
        "data_root": str(data_root),
        "faiss_index_path": str(settings.faiss_index_path),
        "bm25_index_path": str(bm25_path),
        "file_count": len(docs),
        "chunk_count": len(chunks),
    }


# 向后兼容旧名称
rebuild_faiss_index = rebuild_index


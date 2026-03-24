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
    从 data/ 目录重新构建全量拉取 Chroma 向量索引 + BM25 倒排索引。
    Chroma 自带幂等性保护，未被修改的文件即使重复插入也不会重复产生 Embedding Token 花销。

    返回简单的统计信息，便于 /knowledge/index 接口展示。
    """
    settings = get_settings()
    data_root = Path(__file__).resolve().parent.parent.parent / "data"

    docs = _load_markdown_docs(data_root)
    chunks = _split_docs(docs)

    if not chunks:
        return {
            "data_root": str(data_root),
            "chroma_persist_dir": str(settings.chroma_persist_dir),
            "file_count": len(docs),
            "chunk_count": 0,
            "warning": "No chunks generated. Check data/ directory.",
        }

    # 1. 插入或覆盖 Chroma 向量索引
    vs = build_vector_store()
    vs.add_documents(chunks)
    # Chroma 自动持久化，无需 vs.save()

    # 2. 同步构建 BM25 倒排索引（持久化到与 Chroma 相同的父目录）
    bm25_path = settings.chroma_persist_dir.parent / "bm25_index.pkl"
    bm25 = BM25Retriever.build(chunks, save_path=bm25_path)  # noqa: F841

    return {
        "data_root": str(data_root),
        "chroma_persist_dir": str(settings.chroma_persist_dir),
        "bm25_index_path": str(bm25_path),
        "file_count": len(docs),
        "chunk_count": len(chunks),
    }

def increment_index(files: List[Path]) -> dict:
    """增量只解析有更新的/新上传的文件，节省时间，同时重全量建内存 BM25 以保持同步"""
    settings = get_settings()
    from langchain_community.document_loaders import TextLoader
    
    new_docs = []
    for file_path in files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        new_docs.extend(loader.load())

    new_chunks = _split_docs(new_docs)

    if new_chunks:
        # Chroma 增量插入
        vs = build_vector_store()
        vs.add_documents(new_chunks)

    # BM25 需要全局重新构建才能索引新增字词频率，但因为不消耗 API 只是本地执行，成本极低
    data_root = Path(__file__).resolve().parent.parent.parent / "data"
    all_local_docs = _load_markdown_docs(data_root)
    all_chunks = _split_docs(all_local_docs)

    bm25_path = settings.chroma_persist_dir.parent / "bm25_index.pkl"
    if all_chunks:
        BM25Retriever.build(all_chunks, save_path=bm25_path)

    return {
        "incremental_chunks": len(new_chunks),
        "total_bm25_chunks": len(all_chunks) if all_chunks else 0,
        "bm25_index_path": str(bm25_path),
    }


# 向后兼容旧名称
rebuild_faiss_index = rebuild_index


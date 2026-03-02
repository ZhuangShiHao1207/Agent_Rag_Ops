"""
文档处理与向量化脚本
实现完整的 NLP 预处理 + 向量检索 Pipeline：
1. 文本清洗（正则、停用词）
2. 智能分块（语义分块）
3. 关键词提取（Jieba + TextRank）
4. 向量化（Sentence Transformers）
5. 混合索引构建（FAISS + BM25）
"""
import os
import json
import re
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import pickle

# NLP 工具
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# 向量检索
import faiss
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()


class NLPPreprocessor:
    """NLP 预处理器：展示传统 NLP 技能"""
    
    def __init__(self):
        # 中文停用词表
        self.stopwords = self._load_stopwords()
        # 初始化 Jieba
        jieba.initialize()
    
    def _load_stopwords(self) -> set:
        """加载停用词表"""
        # 这里使用简化的停用词表，实际可以加载更完整的
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '啊', '呢'
        }
        return stopwords
    
    def clean_text(self, text: str) -> str:
        """文本清洗：去除噪声字符、多余空白"""
        # 去除特殊字符（保留中英文、数字、常用标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：""''（）【】\n]', '', text)
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        # 去除多余换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """提取关键词（使用 Jieba TextRank）"""
        keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=False)
        return list(keywords)
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """分词并去除停用词"""
        words = jieba.lcut(text)
        if remove_stopwords:
            words = [w for w in words if w not in self.stopwords and len(w) > 1]
        return words


class SemanticChunker:
    """语义分块器：按段落和语义边界智能切分"""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_structure(self, text: str, doc_id: str) -> List[Dict]:
        """按文档结构切分（识别章节标题）"""
        chunks = []
        
        # 识别结构化章节（【】包裹的标题）
        sections = re.split(r'(【[^】]+】)', text)
        
        current_section = ""
        char_offset = 0
        chunk_id = 0
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # 如果是标题，开始新段落
            if section.startswith('【') and section.endswith('】'):
                current_section = section
                char_offset += len(section)
                continue
            
            # 内容块
            content = section.strip()
            if not content:
                continue
            
            # 如果内容太长，按句子切分
            if len(content) > self.chunk_size:
                sentences = re.split(r'([。！？\n])', content)
                temp_chunk = ""
                
                for j in range(0, len(sentences), 2):
                    sentence = sentences[j] + (sentences[j+1] if j+1 < len(sentences) else '')
                    
                    if len(temp_chunk) + len(sentence) > self.chunk_size and temp_chunk:
                        # 保存当前 chunk
                        chunks.append({
                            "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                            "text": (current_section + "\n" + temp_chunk).strip(),
                            "char_start": char_offset,
                            "char_end": char_offset + len(temp_chunk),
                            "section": current_section
                        })
                        chunk_id += 1
                        # 重叠部分
                        temp_chunk = temp_chunk[-self.overlap:] + sentence if len(temp_chunk) > self.overlap else sentence
                    else:
                        temp_chunk += sentence
                
                if temp_chunk:
                    chunks.append({
                        "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                        "text": (current_section + "\n" + temp_chunk).strip(),
                        "char_start": char_offset,
                        "char_end": char_offset + len(temp_chunk),
                        "section": current_section
                    })
                    chunk_id += 1
            else:
                # 内容适中，直接作为一个 chunk
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                    "text": (current_section + "\n" + content).strip(),
                    "char_start": char_offset,
                    "char_end": char_offset + len(content),
                    "section": current_section
                })
                chunk_id += 1
            
            char_offset += len(section)
        
        return chunks


class HybridIndexBuilder:
    """混合索引构建器：FAISS（向量） + BM25（关键词）"""
    
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-zh-v1.5"):
        print(f"加载 Embedding 模型: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        self.nlp_processor = NLPPreprocessor()
        
        # FAISS 索引
        self.faiss_index = None
        # BM25 索引
        self.bm25_index = None
        self.bm25_corpus = []
        
        # 元数据存储
        self.chunk_metadata = []
    
    def build_index(self, chunks: List[Dict], doc_metadata: Dict):
        """构建混合索引"""
        print(f"处理 {len(chunks)} 个文本块...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        # 1. 构建向量索引
        print("生成向量...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # 创建 FAISS 索引
        self.faiss_index = faiss.IndexFlatIP(self.dimension)  # 内积（余弦相似度）
        self.faiss_index.add(embeddings.astype('float32'))
        
        # 2. 构建 BM25 索引
        print("构建 BM25 索引...")
        tokenized_corpus = [self.nlp_processor.tokenize(text) for text in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = tokenized_corpus
        
        # 3. 保存元数据（为每个 chunk 添加关键词）
        for i, chunk in enumerate(chunks):
            keywords = self.nlp_processor.extract_keywords(chunk['text'], top_k=5)
            self.chunk_metadata.append({
                **chunk,
                "doc_id": doc_metadata['doc_id'],
                "doc_title": doc_metadata['title'],
                "category": doc_metadata['category'],
                "keywords": keywords,
                "embedding_index": i
            })
        
        print(f"✅ 索引构建完成：{len(self.chunk_metadata)} 个块")
    
    def save_index(self, output_dir: str):
        """保存索引到磁盘"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 FAISS 索引
        faiss.write_index(self.faiss_index, os.path.join(output_dir, "faiss.index"))
        
        # 保存 BM25 和元数据
        with open(os.path.join(output_dir, "bm25_index.pkl"), 'wb') as f:
            pickle.dump({
                'bm25': self.bm25_index,
                'corpus': self.bm25_corpus
            }, f)
        
        with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 索引已保存到: {output_dir}")


def process_all_documents(
    docs_dir: str = "./data/docs",
    output_dir: str = "./data/indexes"
):
    """处理所有文档并构建索引"""
    
    # 加载所有文档
    all_docs_path = os.path.join(docs_dir, "_all_docs.json")
    if not os.path.exists(all_docs_path):
        print(f"❌ 未找到文档汇总文件: {all_docs_path}")
        print("请先运行 generate_docs.py 生成文档")
        return
    
    with open(all_docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"📚 加载了 {len(documents)} 份文档")
    
    # 初始化处理器
    preprocessor = NLPPreprocessor()
    chunker = SemanticChunker(chunk_size=600, overlap=100)
    index_builder = HybridIndexBuilder()
    
    # 处理每个文档
    all_chunks = []
    for doc in tqdm(documents, desc="处理文档"):
        # 清洗文本
        cleaned_text = preprocessor.clean_text(doc['content'])
        
        # 分块
        chunks = chunker.chunk_by_structure(cleaned_text, doc['doc_id'])
        
        # 构建索引
        index_builder.build_index(chunks, doc)
        
        all_chunks.extend(chunks)
    
    # 保存索引
    index_builder.save_index(output_dir)
    
    print(f"\n📊 处理统计:")
    print(f"   - 文档数: {len(documents)}")
    print(f"   - 文本块数: {len(all_chunks)}")
    print(f"   - 平均每文档: {len(all_chunks)/len(documents):.1f} 块")


if __name__ == "__main__":
    process_all_documents()

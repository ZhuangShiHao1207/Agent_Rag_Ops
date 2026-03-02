"""
评测数据集构建脚本
基于生成的文档，创建高质量的 QA 测试集
"""
import os
import json
import sys
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# 导入统一的 LLM 适配器
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_engine.llm import get_llm_client, get_default_model

client = get_llm_client()

# QA 生成的 Prompt
QA_GENERATION_PROMPT = """你是一位专业的测试数据构建专家。请基于以下企业文档内容，生成 5 个高质量的问答对。

**文档标题**: {title}
**文档内容**:
{content}

**要求**:
1. 问题类型多样化：
   - 2个事实型问题（直接查找答案）
   - 2个流程型问题（需要理解步骤）
   - 1个推理型问题（需要结合上下文）

2. 答案要求：
   - 必须能从文档中找到明确依据
   - 答案长度：20-100字
   - 使用文档中的原词

3. 输出格式（JSON）：
```json
[
  {{
    "question": "具体的问题？",
    "answer": "简洁的答案",
    "difficulty": "easy/medium/hard",
    "type": "factual/procedural/reasoning"
  }}
]
```

请直接返回 JSON 数组，不要添加任何解释。
"""


def generate_qa_for_document(doc: Dict) -> List[Dict]:
    """为单个文档生成 QA 对"""
    
    prompt = QA_GENERATION_PROMPT.format(
        title=doc['title'],
        content=doc['content'][:2000]  # 限制长度避免超 token
    )
    
    try:
        response = client.chat.completions.create(
            model=get_default_model(),
            messages=[
                {"role": "system", "content": "你是测试数据生成专家，输出严格的 JSON 格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # 提取 JSON（去除可能的 markdown 标记）
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        qa_pairs = json.loads(content)
        
        # 为每个 QA 添加文档引用
        for qa in qa_pairs:
            qa['doc_id'] = doc['doc_id']
            qa['doc_title'] = doc['title']
            qa['category'] = doc['category']
        
        return qa_pairs
        
    except Exception as e:
        print(f"生成 QA 失败 [{doc['title']}]: {e}")
        return []


def build_eval_set(
    docs_path: str = "./data/docs/_all_docs.json",
    output_path: str = "./data/eval/golden_qa.json",
    sample_size: int = 15
):
    """构建评测数据集"""
    
    # 加载文档
    with open(docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"📚 加载了 {len(documents)} 份文档")
    
    # 采样（避免生成太多）
    import random
    sampled_docs = random.sample(documents, min(sample_size, len(documents)))
    
    all_qa = []
    
    for doc in tqdm(sampled_docs, desc="生成 QA 数据"):
        qa_pairs = generate_qa_for_document(doc)
        all_qa.extend(qa_pairs)
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 生成了 {len(all_qa)} 条 QA 数据")
    print(f"💾 保存到: {output_path}")
    
    # 统计
    difficulty_count = {}
    type_count = {}
    for qa in all_qa:
        difficulty_count[qa.get('difficulty', 'unknown')] = difficulty_count.get(qa.get('difficulty', 'unknown'), 0) + 1
        type_count[qa.get('type', 'unknown')] = type_count.get(qa.get('type', 'unknown'), 0) + 1
    
    print(f"\n📊 统计:")
    print(f"   难度分布: {difficulty_count}")
    print(f"   类型分布: {type_count}")


if __name__ == "__main__":
    build_eval_set()

"""
增强版评测数据集构建脚本
专门生成高难度、复杂场景的 QA 对，包括：
1. 多跳推理（需要跨文档/跨章节）
2. 模糊指代（需要理解上下文）
3. 并列流程（多个选项/分支）
4. 数字/额度/权限范围（精确匹配）
5. 条款冲突（需要综合判断）
6. 条件判断（if-else 逻辑）
"""
import os
import json
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import random

load_dotenv()

# 导入 LLM 适配器
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_engine.llm.llm_adapter import get_llm_client

client = get_llm_client()

# 复杂场景的 Prompt 模板
ADVANCED_QA_PROMPTS = {
    "multi_hop": """基于以下企业文档，生成 3 个**多跳推理问题**（需要结合多个章节或多条信息才能回答）：

**文档标题**: {title}
**文档内容**: {content}

**要求**:
1. 问题必须需要综合至少 2-3 个不同段落/条款的信息
2. 不能通过简单搜索一句话就能回答
3. 需要推理或计算

**输出格式（JSON）**:
```json
[
  {{
    "question": "如果员工既要申请年假又涉及差旅，需要经过哪些审批流程？",
    "answer": "需要先在HR系统提交年假申请，获主管批准后，再在财务系统提交差旅预算...",
    "difficulty": "hard",
    "type": "multi_hop",
    "reasoning_steps": ["步骤1：找到年假流程", "步骤2：找到差旅流程", "步骤3：综合两者"]
  }}
]
```

直接返回 JSON，不要其他说明。
""",

    "ambiguous_reference": """基于以下文档，生成 2 个**包含模糊指代**的问题（如"这个"、"上述"、"该情况"等）：

**文档**: {title}
**内容**: {content}

**要求**:
1. 问题中包含指代词，需要结合上下文理解
2. 答案需要先解释指代对象，再给出结果

**输出格式**:
```json
[
  {{
    "question": "上述流程中提到的'3个工作日'是指哪个环节？",
    "answer": "'3个工作日'指的是主管审批环节，从提交申请到审批完成的时限",
    "difficulty": "medium",
    "type": "ambiguous_reference",
    "ambiguous_term": "上述流程"
  }}
]
```
""",

    "parallel_process": """基于以下文档，生成 2 个**并列流程/多选项**问题：

**文档**: {title}
**内容**: {content}

**要求**:
1. 问题涉及"分情况讨论"或"有多种方式"
2. 答案需要列出所有可能的路径

**输出格式**:
```json
[
  {{
    "question": "员工可以通过哪些方式提交报销申请？",
    "answer": "有三种方式：1) OA系统在线提交 2) 邮件发送至财务部 3) 纸质单据送至财务窗口",
    "difficulty": "medium",
    "type": "parallel_process",
    "num_options": 3
  }}
]
```
""",

    "numerical_constraint": """基于以下文档，生成 2 个**涉及数字、额度、权限范围**的精确问题：

**文档**: {title}
**内容**: {content}

**要求**:
1. 问题必须涉及具体数字（金额、天数、人数等）
2. 答案必须精确，不能模糊

**输出格式**:
```json
[
  {{
    "question": "经理级别员工的差旅住宿标准上限是多少？",
    "answer": "经理级别员工一线城市住宿标准为每晚不超过800元，二线城市不超过500元",
    "difficulty": "easy",
    "type": "numerical_constraint",
    "key_numbers": ["800", "500"]
  }}
]
```
""",

    "conditional_logic": """基于以下文档，生成 2 个**条件判断（if-else）**问题：

**文档**: {title}
**内容**: {content}

**要求**:
1. 问题包含条件假设（"如果...那么..."）
2. 答案需要根据条件给出不同结果

**输出格式**:
```json
[
  {{
    "question": "如果发票丢失但有支付凭证，能否报销？需要什么额外手续？",
    "answer": "可以报销，但需要：1)提供银行转账记录或支付宝/微信支付截图 2)填写《发票遗失说明》并由主管签字 3)金额超过1000元需财务经理审批",
    "difficulty": "hard",
    "type": "conditional_logic",
    "condition": "发票丢失但有支付凭证"
  }}
]
```
"""
}


def generate_advanced_qa(doc: Dict, qa_type: str) -> List[Dict]:
    """生成特定类型的复杂 QA"""
    
    prompt_template = ADVANCED_QA_PROMPTS.get(qa_type)
    if not prompt_template:
        return []
    
    prompt = prompt_template.format(
        title=doc['title'],
        content=doc['content'][:2500]  # 增加上下文长度
    )
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "你是企业文档测试专家，擅长构造复杂的测试用例。输出严格的 JSON 格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,  # 提高创造性
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        qa_pairs = json.loads(content)
        
        # 添加文档引用
        for qa in qa_pairs:
            qa['doc_id'] = doc['doc_id']
            qa['doc_title'] = doc['title']
            qa['category'] = doc['category']
        
        return qa_pairs
        
    except Exception as e:
        print(f"生成失败 [{doc['title']} - {qa_type}]: {e}")
        return []


def build_advanced_eval_set(
    docs_path: str = "./data/docs/_all_docs.json",
    output_path: str = "./data/eval/advanced_qa.json",
    sample_size: int = 10
):
    """构建高难度评测集"""
    
    with open(docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"📚 加载了 {len(documents)} 份文档")
    
    # 采样
    sampled_docs = random.sample(documents, min(sample_size, len(documents)))
    
    all_qa = []
    qa_types = list(ADVANCED_QA_PROMPTS.keys())
    
    for doc in tqdm(sampled_docs, desc="生成高难度 QA"):
        # 为每个文档随机选择 2-3 种问题类型
        selected_types = random.sample(qa_types, k=min(3, len(qa_types)))
        
        for qa_type in selected_types:
            qa_pairs = generate_advanced_qa(doc, qa_type)
            all_qa.extend(qa_pairs)
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 生成了 {len(all_qa)} 条高难度 QA")
    print(f"💾 保存到: {output_path}")
    
    # 统计
    type_count = {}
    difficulty_count = {}
    for qa in all_qa:
        qa_type = qa.get('type', 'unknown')
        difficulty = qa.get('difficulty', 'unknown')
        type_count[qa_type] = type_count.get(qa_type, 0) + 1
        difficulty_count[difficulty] = difficulty_count.get(difficulty, 0) + 1
    
    print(f"\n📊 统计:")
    print(f"   类型分布: {type_count}")
    print(f"   难度分布: {difficulty_count}")
    
    return all_qa


def merge_eval_sets(
    basic_path: str = "./data/eval/golden_qa.json",
    advanced_path: str = "./data/eval/advanced_qa.json",
    output_path: str = "./data/eval/complete_qa.json"
):
    """合并基础和高级评测集"""
    
    all_qa = []
    
    if os.path.exists(basic_path):
        with open(basic_path, 'r', encoding='utf-8') as f:
            all_qa.extend(json.load(f))
    
    if os.path.exists(advanced_path):
        with open(advanced_path, 'r', encoding='utf-8') as f:
            all_qa.extend(json.load(f))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 合并完成，共 {len(all_qa)} 条 QA")
    print(f"💾 保存到: {output_path}")


if __name__ == "__main__":
    # 生成高难度评测集
    build_advanced_eval_set(sample_size=10)
    
    # 合并基础和高级评测集
    merge_eval_sets()

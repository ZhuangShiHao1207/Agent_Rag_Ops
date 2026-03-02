"""
企业文档生成脚本
使用 LLM 生成结构化的企业知识文档（HR、IT、财务等领域）
"""
import os
import json
import sys
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# 导入统一的 LLM 适配器
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_engine.llm import get_llm_client, get_default_model

# 初始化 LLM 客户端（自动根据 LLM_PROVIDER 选择）
client = get_llm_client()

# 文档模板定义
DOCUMENT_TEMPLATES = {
    "HR": [
        "年假申请流程",
        "远程办公政策",
        "试用期考核标准",
        "员工入职指南",
        "绩效评估流程",
        "离职办理流程",
        "请假管理制度",
    ],
    "IT": [
        "VPN使用说明",
        "密码重置流程",
        "办公设备申领SOP",
        "软件安装权限申请",
        "网络故障报修指南",
        "邮箱配置教程",
        "视频会议系统使用",
    ],
    "Finance": [
        "差旅报销指南",
        "发票提交规范",
        "预算申请流程",
        "采购审批流程",
        "费用报销时效要求",
        "固定资产管理",
    ],
    "CustomerService": [
        "客户投诉处理流程",
        "SLA服务等级协议",
        "退款政策说明",
        "工单升级机制",
        "客户信息保密规定",
    ],
    "Security": [
        "数据隐私保护政策",
        "内部审计流程",
        "安全事件上报机制",
        "访客管理制度",
    ]
}

# 生成文档的 Prompt 模板
GENERATION_PROMPT = """你是一位专业的企业制度文档编写专家。请根据以下要求生成一份结构化的企业文档：

**文档主题**: {title}
**所属领域**: {category}

**要求**：
1. 文档必须包含以下结构化章节：
   - 【目的】：说明该文档的用途
   - 【适用范围】：明确哪些人员/部门适用
   - 【详细流程/规则】：具体的步骤或规则（使用编号列表）
   - 【注意事项】：关键提醒
   - 【常见问题】：3-5个FAQ

2. 语言要求：
   - 正式、专业、无歧义
   - 使用明确的动词（如"提交"、"审批"、"通知"）
   - 避免模糊表述（如"可能"、"大概"）

3. 长度要求：500-800字

4. 必须包含具体细节：
   - 时间期限（如"3个工作日内"）
   - 责任人（如"直属主管"、"IT部门"）
   - 平台/系统名称（如"OA系统"、"HR系统"）

请直接输出文档内容，不要添加任何额外说明。
"""


def generate_document(title: str, category: str) -> Dict:
    """使用 LLM 生成单个企业文档"""
    
    prompt = GENERATION_PROMPT.format(title=title, category=category)
    
    try:
        response = client.chat.completions.create(
            model=get_default_model(),
            messages=[
                {"role": "system", "content": "你是企业文档生成助手，擅长编写结构化的规章制度。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # 构建文档对象
        doc = {
            "doc_id": f"doc_{category.lower()}_{len(content) % 1000:03d}",
            "title": title,
            "category": category,
            "source_type": "synthetic_llm",
            "content": content,
            "word_count": len(content),
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "model": get_default_model(),
                "temperature": 0.7
            }
        }
        
        return doc
        
    except Exception as e:
        print(f"生成文档失败 [{title}]: {e}")
        return None


def generate_all_documents(output_dir: str = "./data/docs", num_per_category: int = None):
    """批量生成所有文档"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_docs = []
    total = sum(len(titles) if num_per_category is None else min(len(titles), num_per_category) 
                for titles in DOCUMENT_TEMPLATES.values())
    
    with tqdm(total=total, desc="生成企业文档") as pbar:
        for category, titles in DOCUMENT_TEMPLATES.items():
            # 如果指定了每类数量，则截取
            if num_per_category:
                titles = titles[:num_per_category]
            
            for title in titles:
                doc = generate_document(title, category)
                if doc:
                    all_docs.append(doc)
                    
                    # 保存单个文档
                    filename = f"{doc['doc_id']}.json"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(doc, f, ensure_ascii=False, indent=2)
                
                pbar.update(1)
    
    # 保存汇总文件
    summary_path = os.path.join(output_dir, "_all_docs.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 成功生成 {len(all_docs)} 份文档，保存在: {output_dir}")
    print(f"📊 统计:")
    for category in DOCUMENT_TEMPLATES.keys():
        count = sum(1 for d in all_docs if d['category'] == category)
        print(f"   - {category}: {count} 份")
    
    return all_docs


if __name__ == "__main__":
    # 生成所有文档（可以通过参数控制每类生成数量）
    # generate_all_documents(num_per_category=1)  # 每类生成1份，测试
    generate_all_documents()  # 生成全部


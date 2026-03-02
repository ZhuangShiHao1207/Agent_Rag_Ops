"""
统一的 LLM 适配器
支持多种 LLM 提供商：OpenAI、混元（Hunyuan）等
通过配置文件或环境变量切换
"""
import os
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMAdapter:
    """LLM 适配器基类"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.client = self._create_client(**kwargs)
    
    def _create_client(self, **kwargs) -> OpenAI:
        """创建客户端"""
        if self.base_url:
            return OpenAI(api_key=self.api_key, base_url=self.base_url, **kwargs)
        else:
            return OpenAI(api_key=self.api_key, **kwargs)
    
    def chat_completion(self, model: str, messages: list, **kwargs) -> Any:
        """对话补全"""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
    
    def embeddings(self, model: str, input_text: str, **kwargs) -> Any:
        """生成 Embedding"""
        return self.client.embeddings.create(
            model=model,
            input=input_text,
            **kwargs
        )


class OpenAIAdapter(LLMAdapter):
    """OpenAI 适配器"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
    
    def chat_completion(self, model: str, messages: list, **kwargs) -> Any:
        """OpenAI 对话（标准实现）"""
        return super().chat_completion(model, messages, **kwargs)
    
    def embeddings(self, model: str, input_text: str, **kwargs) -> Any:
        """OpenAI Embedding（标准实现）"""
        return super().embeddings(model, input_text, **kwargs)


class HunyuanAdapter(LLMAdapter):
    """腾讯混元适配器"""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("HUNYUAN_API_KEY")
        base_url = "https://api.hunyuan.cloud.tencent.com/v1"
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
    
    def chat_completion(self, model: str, messages: list, **kwargs) -> Any:
        """混元对话（兼容 OpenAI 格式）"""
        # 混元特有参数处理
        hunyuan_kwargs = kwargs.copy()
        
        # 可选：添加混元特有参数
        if 'enable_enhancement' not in hunyuan_kwargs:
            hunyuan_kwargs['enable_enhancement'] = True
        
        return super().chat_completion(model, messages, **hunyuan_kwargs)
    
    def embeddings(self, model: str, input_text: str, **kwargs) -> Any:
        """混元 Embedding
        注意：混元 embedding 固定 1024 维，模型名固定为 hunyuan-embedding
        """
        # 强制使用混元的 embedding 模型
        hunyuan_model = "hunyuan-embedding"
        
        return super().embeddings(hunyuan_model, input_text, **kwargs)
    
    def get_embedding_dimension(self) -> int:
        """返回混元 embedding 维度（固定 1024）"""
        return 1024


def get_llm_client(provider: Optional[str] = None) -> OpenAI:
    """
    获取 LLM 客户端（工厂方法）
    
    Args:
        provider: LLM 提供商 ('openai', 'hunyuan')，默认从环境变量读取
    
    Returns:
        OpenAI 兼容的客户端对象
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        adapter = OpenAIAdapter()
    elif provider == "hunyuan":
        adapter = HunyuanAdapter()
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
    
    return adapter.client


def get_llm_adapter(provider: Optional[str] = None) -> LLMAdapter:
    """
    获取 LLM 适配器（高级用法，可访问适配器特有方法）
    
    Args:
        provider: LLM 提供商
    
    Returns:
        LLMAdapter 对象
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        return OpenAIAdapter()
    elif provider == "hunyuan":
        return HunyuanAdapter()
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")


# 便捷函数：获取当前配置的模型名称
def get_default_model() -> str:
    """根据提供商返回默认模型"""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        return os.getenv("LLM_MODEL", "gpt-4o-mini")
    elif provider == "hunyuan":
        return os.getenv("LLM_MODEL", "hunyuan-turbo")
    else:
        return "gpt-4o-mini"


def get_embedding_model() -> str:
    """根据提供商返回默认 Embedding 模型"""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    elif provider == "hunyuan":
        return "hunyuan-embedding"  # 混元固定模型名
    else:
        return "text-embedding-3-small"


def get_embedding_dimension() -> int:
    """根据提供商返回 Embedding 维度"""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "openai":
        # OpenAI text-embedding-3-small 是 1536 维
        return int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    elif provider == "hunyuan":
        return 1024  # 混元固定 1024 维
    else:
        return 1536


if __name__ == "__main__":
    # 测试代码
    print("当前 LLM 配置:")
    print(f"  提供商: {os.getenv('LLM_PROVIDER', 'openai')}")
    print(f"  对话模型: {get_default_model()}")
    print(f"  Embedding 模型: {get_embedding_model()}")
    print(f"  Embedding 维度: {get_embedding_dimension()}")
    
    # 测试客户端
    try:
        client = get_llm_client()
        print("\n✅ LLM 客户端初始化成功")
    except Exception as e:
        print(f"\n❌ LLM 客户端初始化失败: {e}")

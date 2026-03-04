from typing import Any, List

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.config import get_settings

HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"


class LLMClient(Embeddings):
    """
    统一封装 LLM 能力，并实现 LangChain `Embeddings` 接口，
    以便直接被 FAISS 等向量库使用。

    支持：
    - OpenAI（直接使用官方 SDK）
    - Hunyuan（混元）：OpenAI 兼容接口，切换 api_key + base_url 即可
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.llm_provider

        if self.provider == "openai":
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_api_base,
            )
        elif self.provider == "hunyuan":
            if not self.settings.hunyuan_api_key:
                raise ValueError(
                    "HUNYUAN_API_KEY is not set. "
                    "Please add it to your .env file: HUNYUAN_API_KEY=your-key"
                )
            self._client = OpenAI(
                api_key=self.settings.hunyuan_api_key,
                base_url=self.settings.hunyuan_api_base,
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

    # ===== 对话能力 =====

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """
        统一对话接口，Hunyuan 和 OpenAI 均通过 OpenAI SDK 兼容层调用。
        """
        response = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.choices[0].message.content or ""

    # ===== Embeddings 接口（供 LangChain 使用）=====

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain Embeddings 接口：对一批文档进行向量化。
        Hunyuan 和 OpenAI 均通过 OpenAI SDK 兼容层调用。
        """
        resp = self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        """
        LangChain Embeddings 接口：对单条查询进行向量化。
        """
        return self.embed_documents([text])[0]


def build_llm_client() -> LLMClient:
    return LLMClient()


def build_langchain_llm(**kwargs):
    """
    返回与 LangGraph 事件系统兼容的 LangChain ChatOpenAI 实例。
    使用此函数替代 build_llm_client() 可正确触发 on_chat_model_stream 事件，
    从而使 SSE 流式输出正常工作。
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or "https://api.openai.com/v1",
            **kwargs,
        )
    elif settings.llm_provider == "hunyuan":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.hunyuan_api_key,
            base_url=settings.hunyuan_api_base,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


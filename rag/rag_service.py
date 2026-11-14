
import os

from openai import OpenAI
from dotenv import load_dotenv
from rag.prompt_builder import PromptBuilder
from rag.reranker import get_reranker
load_dotenv()
class RAGService:
    """
    协调：
    Retrieval -> Prompt -> LLM -> Answer
    """

    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service
        self.prompt_builder = PromptBuilder()
        # api_key = os.getenv("LLM_API_KEY")
        api_key = os.environ["LLM_API_KEY"]
        if not api_key:
            raise RuntimeError(
                "缺少 LLM_API_KEY 环境变量"
            )

        # base_url = os.getenv("LLM_BASE_URL")
        base_url = os.environ["LLM_BASE_URL"]
        if not base_url:
            raise RuntimeError(
                "缺少 LLM_BASE_URL 环境变量"
            )

        # self.model = os.getenv(
        #     "LLM_MODEL", "qwen3.8-flash"
        # )
        self.model = os.environ["LLM_MODEL"]

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=1,
        )

    def ask(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 3,
        scope: str = "body",
        rerank: bool = False,
    ) -> dict:

        # 第一阶段：检索论文证据
        retrieval = self.retrieval_service.search(
            query=query,
            method=method,
            top_k=top_k,
            scope=scope,
            rerank = rerank,
        )

        sources = retrieval["results"]

        # 没有证据就不调用LLM
        if not sources:
            return {
                "query": query,
                "answer": "没有检索到可用的论文证据，无法可靠回答。",
                "sources": [],
                "model": None,
            }

        # 第二阶段：构建Prompt
        messages = self.prompt_builder.build_messages(
            query=query,
            results=sources,
        )

        # 第三阶段：调用通义千问
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )

        answer = completion.choices[0].message.content

        # 第四阶段：返回答案和实际检索证据
        return {
            "query": query,
            "answer": answer or "",
            "sources": sources,
            "model": completion.model,
        }

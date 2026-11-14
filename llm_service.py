
import os

from dotenv import load_dotenv
from openai import OpenAI,AsyncOpenAI

# 读取项目的 .env 文件
load_dotenv()

# 初始化 LLM API 客户端
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    timeout=90.0,
    max_retries=1,
)

#初始化异步客户端
async_client = AsyncOpenAI(
      api_key = os.environ["LLM_API_KEY"],
      base_url = os.environ["LLM_BASE_URL"],
      timeout = 90.0,
      max_retries = 0,
)

def ask_llm(question: str, system_prompt:str = "你是一名专业的AI技术助手,请使用中文回答。") -> str:
    """
    调用大模型，返回生成的文本答案。
    """
    #普通调用，这里的answer是一个完整响应对象
    response = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[
            {
                "role": "system",
                # "content": "请像大学老师一样详细解释"
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens = 500,
        # stream = True
        #设置为True是流式输出，设置为Fasle是非流式输出
    )
    answer = response.choices[0].message.content
    return answer

def stream_llm(question:str,system_prompt:str = "你是一名专业的AI技术助手，请使用中文回答。"):
    """流式调用大模型，逐步返回生成的文本，"""

    stream = client.chat.completions.create(
            model = os.environ["LLM_MODEL"],
            messages = [
                    {
                            "role":"system",
                            "content":system_prompt
                    },
                    {
                            "role":"user",
                            "content":question
                    }
            ],
            stream = True,
            max_tokens = 1000,
    )
    try:
        for chunk in stream:
                #先检查是否存在choices
                if not chunk.choices:
                        continue

                choice = chunk.choices[0]

                #提取当前生成的文本
                content = choice.delta.content

                if content:
                        yield content
    finally:
            stream.close()
    
async def ask_llm_async(
    question: str,
    system_prompt: str = "你是一名专业的AI技术助手，请使用中文回答。"
) -> str:

    response = await async_client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=500,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("模型没有返回有效文本")

    return answer

async def stream_llm_async(question: str, system_prompt: str = "你是一名专业的AI技术助手，请使用中文回答。"):
      """
      异步调用LLM，并逐步产生文本片段
      """
      #1，等待模型简历流式响应
      stream = await async_client.chat.completions.create(
            model = os.environ["LLM_MODEL"],
            messages = [
                  {
                        "role" : "system",
                        "content": system_prompt
                  },
                  {
                        "role":"user",
                        "content":question
                  }
            ],
            stream = True,
            max_tokens = 500,
      )
      try:
            #2异步读取数据模型返回的数据块
            async for chunk in stream:
                  #部分chunk可能不包含stream:
                  if not chunk.choices:
                        continue
                  choice = chunk.choices[0]
                  content = choice.delta.content

                  #逐步返回有效文本
                  if content:
                        yield content
      finally:
            #4.关闭流式传输
            await stream.close()
"""
第一次等待：等待模型服务建立响应，拿到可以持续读取的流。

后续等待：每次读取下一个 Chunk 时，如果网络数据尚未到达，就可以让事件循环处理其他任务。

整个过程类似：
向 LLM API 发起请求

await 等待流式响应建立

async for 接收 Chunk 1

yield 输出 Chunk 1

async for 等待 Chunk 2

yield 输出 Chunk 2

在支持异步的网络等待期间，事件循环可以调度其他请求。

注意这里的 yield 只是把数据交给外层响应生成器，是否立即通过网络送达还要经过 FastAPI、ASGI 服务器和客户端等环节。
"""
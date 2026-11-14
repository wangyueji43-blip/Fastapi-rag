
import logging
import json
from fastapi.responses import StreamingResponse
from api.rag_routes import router as rag_router
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

from llm_service import ask_llm,stream_llm,ask_llm_async,stream_llm_async

# 日志
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="FastAPI RAG Agent",
    description="基于 FastAPI 的 AI 应用学习项目",
    version="0.1.0",
)

app.include_router(rag_router)

# 定义用户请求的数据结构
class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
        description="用户的问题",
    )
    system_prompt: str = Field(
        default="你是一名专业的AI技术助手,请使用中文回答。",
        min_length = 1,
        max_length = 1000
    )


# 健康检查
@app.get("/health")
def health():
    return {"status": "ok"}


# 路径参数练习
@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"hello, {name}!"}


# 真实 LLM 问答
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = ask_llm(request.question, request.system_prompt)

        return {
            "question": request.question,
            "answer": answer,
        }

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="模型服务请求超时",
        )

    except APIConnectionError:
        raise HTTPException(
            status_code=502,
            detail="无法连接上游模型服务",
        )

    except APIStatusError as exc:
        logger.warning(
            "LLM API returned status %s",
            exc.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail="上游模型服务调用失败",
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="模型未返回有效答案",
        )

@app.post("/chat/stream")
def chat_stream(request:ChatRequest):

    def generate():
        try:
            #来一个结果取一个结果
            for content in stream_llm(
                question = request.question, 
                system_prompt=request.system_prompt):

                #将文本片段转换为json
                data = json.dumps(
                    {"content":content},
                    ensure_ascii=False
                )

                #按照SSE格式发送
                yield f"data: {data}\n\n"

            yield "data: [DONE]\n\n"
        
        except Exception:
            logger.exception("LLM streaming failed")

            error_data = json.dumps(
                {"message": "模型生成过程中发生错误"},
                ensure_ascii = False
            )

            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
            generate(),
            media_type = "text/event-stream",
            headers = {
                "Cache-control":"no-cache",
                "X-Accel_Buffering":"no",
            }
        )
#这里为什么要加一个新的generate()
    """因为 StreamingResponse 需要一个能够持续提供数据的可迭代对象。
    我们的 stream_llm() 已经是一个生成器，所以这里再封装一层，主要是为了增加 SSE 格式和异常处理
"""

#为什么这里使用json.dumps()
"""模型返回的文本可能包含中文、换行符、引号等特殊字符。

json.dumps() 会负责正确转义这些内容，使它们能够安全地放进 JSON 数据中。

ensure_ascii=False 则让中文保持可读，而不是转换成 XXXX。"""

#为什么使用 yield f"data: {data}\n\n"？
"""这是 SSE 的数据格式。

例如：
data: {"content":"RAG是一种"}
data: {"content":"检索增强生成技术"}
data: [DONE]

每条消息通过两个换行符分隔。"""

#第四：为什么设置 X-Accel-Buffering: no？

"""这个请求头主要用于 Nginx 等反向代理环境，提示代理不要缓存流式响应。

目前我们直接通过 Uvicorn 访问服务器，暂时不依赖它，但保留这个配置有利于以后部署。"""

#这里的使用的新的调控指令
"""curl -N -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请详细介绍一下RAG的工作流程",
    "system_prompt": "你是一名大学老师，请分步骤详细解释"
  }' """
"""
这里的 -N 非常重要：
-N
它会关闭 curl 的输出缓冲，让你能够观察数据逐步到达的过程。"""

@app.post("/chat/async")
async def chat_async(request: ChatRequest):
    try:
        answer = await ask_llm_async(
            question=request.question,
            system_prompt=request.system_prompt,
        )

        return {
            "question": request.question,
            "answer": answer,
        }

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="模型服务请求超时",
        )

    except APIConnectionError:
        raise HTTPException(
            status_code=502,
            detail="无法连接模型服务",
        )

    except APIStatusError as exc:
        logger.warning(
            "Upstream API returned %s",
            exc.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail="模型服务调用失败",
        )

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="模型未返回有效文本",
        )
    
"""
3. 为什么不能简单把 def 改成 async def？
例如下面这种写法就不推荐：
@app.post("/chat/async")
async def chat_async(request: ChatRequest):
    answer = ask_llm(request.question)
    return {"answer": answer}


这里虽然写了 async def，但 ask_llm() 仍然使用同步网络客户端。

它可能阻塞事件循环。

FastAPI 官方文档也明确区分了两类情况：普通 def 路由会在线程池中执行，而 async def 更适合搭配真正支持 await 的异步 I/O 库。
FastAPI


因此我们的正确调用链应该是：
async def chat_async()
        ↓
await ask_llm_async()
        ↓
await AsyncOpenAI API
        ↓
千问模型


而不是只在函数声明前加一个 async。
"""
@app.post("/chat/async-stream")
async def chat_async_stream(request: ChatRequest):
    async def generate():
        try:
            async for content in stream_llm_async(question = request.question,system_prompt = request.system_prompt,):
                data = json.dumps({"content":content},
                                  ensure_ascii= False)
                yield f"data:{data}\n\n"
        except Exception:
            logger.exception("Async LLM streaming fialed")
            error_data = json.dump(
                {"message":"模型流式生成失败"},
                ensure_ascii=False
            )
            yield (
                f"event: error\n"
                f"data: {error_data}\n\n"
            )
    return StreamingResponse(
        generate(), media_type = "text/event-stream",
        headers = {"Cache-Control" : "no-cache",
                   "X-Accel-Buffering": "no",},
    )

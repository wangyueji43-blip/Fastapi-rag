
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from rag.retrieval_service import RetrievalService
from schemas.rag import SearchRequest, SearchResponse

from fastapi import HTTPException, Depends

from rag.rag_service import RAGService
from schemas.rag import AskRequest, AskResponse

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """
    复用同一个RetrievalService实例。
    """
    return RetrievalService()


@router.post(
    "/search",
    response_model=SearchResponse,
)
def rag_search(
    request: SearchRequest,
    service: RetrievalService = Depends(
        get_retrieval_service
    ),
):  
    print(
    "[DEBUG]",
    "rerank =", request.rerank,
    "type =", type(request.rerank),
    flush=True,
    )
    try:
        return service.search(
            query=request.query,
            method=request.method,
            top_k=request.top_k,
            scope=request.scope,
            rerank=request.rerank,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post(
    "/ask",
    response_model=AskResponse,
)
def rag_ask(
    request: AskRequest,
    retrieval_service=Depends(get_retrieval_service),
):  
    print(
        "[DEBUG]",
        "rerank =", request.rerank,
        "type =", type(request.rerank),
        flush=True,
        )
    try:
        # 复用已有RetrievalService
        rag_service = RAGService(
            retrieval_service=retrieval_service
        )

        result = rag_service.ask(
            query=request.query,
            method=request.method,
            top_k=request.top_k,
            scope=request.scope,
            rerank =request.rerank,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

### 这里的数据流值得特别注意

# 你之前已经学习过 `Depends()`，现在它又发挥作用了：
"""
```python
retrieval_service=Depends(get_retrieval_service)
```

FastAPI 会先获取我们之前缓存的 `RetrievalService`。

再执行：

```python
rag_service = RAGService(retrieval_service)
```

因此两个 API 共用同一个检索服务对象：

```text
                get_retrieval_service()
                         │
                 RetrievalService
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
        /rag/search              /rag/ask
             │                       │
        直接返回证据             RAGService
                                     │
                                PromptBuilder
                                     │
                                  通义千问
                                     │
                                返回生成答案
```
"""
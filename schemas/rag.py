
from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="用户查询问题",
    )

    method: Literal[
        "dense", "bm25", "hybrid"
    ] = "hybrid"

    scope: Literal[
        "body", "reference", "all"
    ] = "body"

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="返回的检索结果数量",
    )
    rerank :bool = False


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    source: str
    content: str
    pages: list[int]
    headings: list[str]
    content_type: str
    score: float
    score_type: str
    # rerank:bool = False


class SearchResponse(BaseModel):
    query: str
    method: str
    scope: str
    total: int
    results: list[SearchResult]


from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1)

    method: Literal[
        "dense", "bm25", "hybrid"
    ] = "hybrid"

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )

    scope: Literal[
        "body", "reference", "all"
    ] = "body"
    
    rerank:bool = False


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchResult]
    model: str | None = None


import pytest

from fastapi.testclient import TestClient

from main import app
from api.rag_routes import get_retrieval_service


class FakeRetrievalService:
    """
    模拟检索服务。

    不加载Embedding模型，也不执行真实检索。
    只返回预先设定的测试数据。
    """

    def __init__(self):
        self.calls = []

    def search(
        self,
        query,
        method="hybrid",
        top_k=5,
        scope="body",
        rerank = False,
    ):
        # 记录API传入的参数
        self.calls.append({
            "query": query,
            "method": method,
            "top_k": top_k,
            "scope": scope,
            "rerank": rerank,
        })

        # 根据scope模拟不同内容类型
        content_type = (
            "reference"
            if scope == "reference"
            else "body"
        )

        results = [
            {
                "rank": 1,
                "chunk_id": "test_001",
                "document_id": "sample",
                "source": "sample.pdf",
                "content": "DCMAGRN consists of four modules.",
                "pages": [2],
                "headings": ["Framework"],
                "content_type": content_type,
                "score": 0.03279,
                "score_type": method,
                "rerank":False,
            }
        ]

        return {
            "query": query,
            "method": method,
            "scope": scope,
            "total": len(results),
            "results": results,
        }


@pytest.fixture
def api_client():
    """
    每个测试创建独立的Fake Service，
    并暂时覆盖真实检索依赖。
    """

    fake_service = FakeRetrievalService()

    app.dependency_overrides[
        get_retrieval_service
    ] = lambda: fake_service

    try:
        with TestClient(app) as client:
            yield client, fake_service
    finally:
        app.dependency_overrides.pop(
            get_retrieval_service, None
        )


def test_search_success(api_client):
    """正常请求应该返回200。"""

    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "query": "DCMAGRN architecture",
            "method": "hybrid",
            "top_k": 3,
            "scope": "body",
            "rerank":False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "DCMAGRN architecture"
    assert data["method"] == "hybrid"
    assert data["total"] == 1

    assert data["results"][0]["chunk_id"] == "test_001"
    assert data["results"][0]["pages"] == [2]

    # 验证Router正确传递参数
    assert fake.calls == [{
        "query": "DCMAGRN architecture",
        "method": "hybrid",
        "top_k": 3,
        "scope": "body",
        "rerank":False,
    }]


@pytest.mark.parametrize(
    "method",
    ["dense", "bm25", "hybrid"],
)
def test_supported_methods(api_client, method):
    """三种检索方式都应允许调用。"""

    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "query": "DCMAGRN",
            "method": method,
        },
    )

    assert response.status_code == 200
    assert fake.calls[-1]["method"] == method


def test_invalid_method(api_client):
    """不支持的method应该返回422。"""

    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "query": "DCMAGRN",
            "method": "invalid",
        },
    )

    assert response.status_code == 422

    # Pydantic应在调用Service前拒绝参数
    assert fake.calls == []


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_invalid_top_k(api_client, top_k):
    """top_k必须在1至50之间。"""

    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "query": "DCMAGRN",
            "top_k": top_k,
        },
    )

    assert response.status_code == 422
    assert fake.calls == []


def test_reference_scope(api_client):
    """检查scope是否正确传递。"""

    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "query": "gene regulatory networks",
            "scope": "reference",
            "rerank":False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["scope"] == "reference"
    assert data["results"][0]["content_type"] == "reference"
    assert fake.calls[-1]["scope"] == "reference"


def test_missing_query(api_client):
    """缺少必填query应该返回422。"""

    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "method": "hybrid",
        },
    )

    assert response.status_code == 422
    assert fake.calls == []
def test_rerank_enabled(api_client):
    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "query": "DCMAGRN",
            "method": "hybrid",
            "rerank":True,
        },
    )

    assert response.status_code == 200
    assert fake.calls[-1]["rerank"] is True


def test_rerank_default_disabled(api_client):
    client, fake = api_client

    response = client.post(
        "/rag/search",
        json={
            "query": "DCMAGRN",
        },
    )

    assert response.status_code == 200
    assert fake.calls[-1]["rerank"] is False
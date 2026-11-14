
from pathlib import Path

from rag.embedding_service import EmbeddingService
from rag.reranker import get_reranker
from rag.index_store import load_index

from rag.docling_retrieval import (
    select_candidates,
    dense_search,
    bm25_search,
)

from rag.docling_fusion import rrf_fusion


class RetrievalService:
    """
    RAG检索服务。

    负责协调Dense、BM25和RRF，并返回标准化结果。
    """

    def __init__(self):
        print("正在初始化 RetrievalService...")

        self.chunks, self.vectors, self.manifest = (
            load_index()
        )

        self.embedding_service = EmbeddingService()

        print(
            f"索引加载完成：{len(self.chunks)}个Chunk，"
            f"向量维度={self.vectors.shape[1]}"
        )


    @staticmethod
    def _union_candidates(*result_lists):
        """
        合并不同检索器的候选结果，并按 chunk_id 去重。
        """
        candidates = {}

        for results in result_lists:
            for item in results:
                chunk_id = item["chunk_id"]

                if chunk_id not in candidates:
                    candidates[chunk_id] = item.copy()

        return list(candidates.values())

    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
        scope: str = "body",
        rerank: bool = False,
    ):
        if not query or not query.strip():
            raise ValueError("query不能为空")

        if top_k < 1:
            raise ValueError("top_k必须大于0")

        if method not in {"dense", "bm25", "hybrid"}:
            raise ValueError("不支持的检索方法")

        if scope not in {"body", "reference", "all"}:
            raise ValueError("不支持的检索范围")


        # 放在 RetrievalService.search() 内，
        # 原有参数校验之后、原有检索流程之前。

        if rerank:
            # 每一路最多召回20条。
            # 如果最终top_k大于20，也应保证候选数不少于top_k。
            candidate_k = max(20, top_k)

            if method == "hybrid":
                dense_data = self.search(
                    query=query,
                    method="dense",
                    top_k=candidate_k,
                    scope=scope,
                    rerank=False,
                )

                bm25_data = self.search(
                    query=query,
                    method="bm25",
                    top_k=candidate_k,
                    scope=scope,
                    rerank=False,
                )

                candidates = self._union_candidates(
                    dense_data["results"],
                    bm25_data["results"],
                )

            elif method in {"dense", "bm25"}:
                data = self.search(
                    query=query,
                    method=method,
                    top_k=candidate_k,
                    scope=scope,
                    rerank=False,
                )

                candidates = data["results"]

            else:
                raise ValueError(
                    f"Unsupported retrieval method: {method}"
                )

            reranker = get_reranker()

            results = reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )

            return {
                "query": query,
                "method": method,
                "scope": scope,
                "total": len(results),
                "results": results,
            }

        # rerank=False:
        # 下面保持原有 Dense/BM25/RRF 逻辑不变。

        candidates = select_candidates(
            self.chunks,
            scope,
        )

        if not candidates:
            return {
                "query": query,
                "method": method,
                "scope": scope,
                "total": 0,
                "results": [],
            }

        # Hybrid召回阶段使用较大的候选池
        pool_size = min(
            len(candidates),
            max(top_k * 4, 20),
        )

        if method == "dense":
            ranked = dense_search(
                query,
                self.embedding_service,
                self.vectors,
                candidates,
                top_k,
            )

            score_type = "cosine_similarity"

        elif method == "bm25":
            ranked = bm25_search(
                query,
                self.chunks,
                candidates,
                top_k,
            )

            score_type = "bm25"

        else:
            dense_results = dense_search(
                query,
                self.embedding_service,
                self.vectors,
                candidates,
                pool_size,
            )

            bm25_results = bm25_search(
                query,
                self.chunks,
                candidates,
                pool_size,
            )

            ranked = rrf_fusion(
                dense_results,
                bm25_results,
                top_k,
            )

            score_type = "rrf"

        formatted = []

        for rank, item in enumerate(ranked, start=1):
            chunk = self.chunks[item["index"]]

            formatted.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "document_id": Path(
                    chunk.get("source", "sample.pdf")
                ).stem,
                "source": chunk.get("source", ""),
                "content": chunk["text"],
                "pages": chunk.get("pages", []),
                "headings": chunk.get("headings", []),
                "content_type": chunk.get(
                    "content_type", "unknown"
                ),
                "score": float(item["score"]),
                "score_type": score_type,
            })

        return {
            "query": query,
            "method": method,
            "scope": scope,
            "total": len(formatted),
            "results": formatted,
        }
    

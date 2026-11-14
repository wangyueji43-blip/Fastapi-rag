
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from rag.retrieval_service import RetrievalService
from rag.rag_service import RAGService

from scripts.test_reranker import (
    MODEL_NAME,
    QUESTIONS,
    build_union,
    rerank,
)


# 暂时只对比最有诊断价值的两道题
TARGET_IDS = {"q002", "q004"}

RETRIEVAL_TOP_K = 20
FINAL_TOP_K = 3

OUTPUT_PATH = Path(
    "data/evaluation/rag_answer_comparison.json"
)


class FixedRetrievalService:
    """
    实验用的检索服务替身。

    不执行新的检索，而是将预先得到的
    RRF 或 Reranker 结果交给 RAGService。
    """

    def __init__(self, result):
        self.result = result

    def search(self, query, method="hybrid",
               top_k=3, scope="body"):
        return self.result


def generate_answer(query, retrieval_result):
    """
    复用已有的 RAGService 和 PromptBuilder。
    只替换检索阶段产生的证据。
    """
    fixed_retriever = FixedRetrievalService(
        retrieval_result
    )

    rag_service = RAGService(
        retrieval_service=fixed_retriever
    )

    return rag_service.ask(
        query=query,
        method="hybrid",
        top_k=FINAL_TOP_K,
        scope="body",
    )


def main():
    print("初始化 RetrievalService...")
    service = RetrievalService()

    print("加载 Cross-Encoder...")
    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
    )

    records = []

    for question in QUESTIONS:
        if question["id"] not in TARGET_IDS:
            continue

        query = question["query"]

        print("\n" + "=" * 70)
        print(question["id"], query)

        # 方案A：当前RRF基线
        baseline = service.search(
            query=query,
            method="hybrid",
            top_k=FINAL_TOP_K,
            scope="body",
        )

        # 方案B：Dense/BM25 Union + Reranker
        dense = service.search(
            query=query,
            method="dense",
            top_k=RETRIEVAL_TOP_K,
            scope="body",
        )

        bm25 = service.search(
            query=query,
            method="bm25",
            top_k=RETRIEVAL_TOP_K,
            scope="body",
        )

        candidates = build_union(
            dense["results"],
            bm25["results"],
        )

        reranked_results = rerank(
            query=query,
            candidates=candidates,
            model=model,
            top_k=FINAL_TOP_K,
        )

        reranker_retrieval = {
            "query": query,
            "method": "hybrid",
            "scope": "body",
            "total": len(reranked_results),
            "results": reranked_results,
        }

        # 保持相同的生成流程
        print("调用通义千问：RRF...")
        baseline_answer = generate_answer(
            query, baseline
        )

        print("调用通义千问：Reranker...")
        reranker_answer = generate_answer(
            query, reranker_retrieval
        )

        print("\n--- RRF Answer ---")
        print(baseline_answer["answer"])

        print("\n--- Reranker Answer ---")
        print(reranker_answer["answer"])

        print("\n--- Source IDs ---")
        print(
            "RRF:",
            [
                item["chunk_id"]
                for item in baseline_answer["sources"]
            ],
        )
        print(
            "Reranker:",
            [
                item["chunk_id"]
                for item in reranker_answer["sources"]
            ],
        )

        records.append({
            "id": question["id"],
            "query": query,
            "rrf": baseline_answer,
            "reranker": reranker_answer,
        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n实验结果已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()

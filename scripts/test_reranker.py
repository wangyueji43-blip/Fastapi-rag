
import time

from sentence_transformers import CrossEncoder

from rag.retrieval_service import RetrievalService


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

RETRIEVAL_TOP_K = 20
FINAL_TOP_K = 3

# 当前开发集中已知的相关证据
# 注意：这只是部分标注，不代表全部正确证据
QUESTIONS = [
    {
        "id": "q001",
        "query": "What are the four main modules of DCMAGRN?",
        "relevant_chunk_id": "docling_0028",
    },
    {
        "id": "q002",
        "query": "Which datasets were used to benchmark DCMAGRN?",
        "relevant_chunk_id": "docling_0018",
    },
    {
        "id": "q003",
        "query": (
            "What roles do the autoencoder and convolutional "
            "neural network play in DCMAGRN?"
        ),
        "relevant_chunk_id": "docling_0041",
    },
    {
        "id": "q004",
        "query": (
            "What do the ablation experiments "
            "demonstrate about DCMAGRN?"
        ),
        "relevant_chunk_id": "docling_0052",
    },
]


def build_union(dense_results, bm25_results):
    """
    将两种检索方法的结果合并、去重。

    用chunk_id作为唯一标识。
    同一个Chunk即使被两路检索命中，也只保留一次。
    """
    candidates = {}

    for item in dense_results + bm25_results:
        chunk_id = item["chunk_id"]

        if chunk_id not in candidates:
            candidates[chunk_id] = item.copy()

    return list(candidates.values())


def rerank(query, candidates, model, top_k=3):
    """
    Cross-Encoder对所有候选重新评分。

    注意：
    这里不能提前把candidates截断到top_k。
    """
    if not candidates:
        return []

    pairs = [
        (query, item["content"])
        for item in candidates
    ]

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked = []

    for item, score in zip(candidates, scores):
        result = item.copy()
        result["score"] = float(score)
        result["score_type"] = "cross_encoder"
        ranked.append(result)

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # 重新生成最终排名
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return ranked[:top_k]


def find_rank(results, target_chunk_id):
    """
    返回某条已知相关证据的排名。

    找不到时返回None。
    """
    for rank, item in enumerate(results, start=1):
        if item["chunk_id"] == target_chunk_id:
            return rank

    return None


def print_results(title, results):
    print(f"\n{title}")

    for item in results:
        print(
            f"  Rank {item['rank']}: "
            f"{item['chunk_id']} | "
            f"score={item['score']:.4f} | "
            f"page={item['pages']}"
        )
        print(
            "    ",
            item["content"][:150].replace("\n", " "),
        )


def main():
    print("初始化 RetrievalService...")
    service = RetrievalService()

    print("\n加载 Cross-Encoder...")
    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
    )

    print("Cross-Encoder 加载完成")

    summary = []

    for question in QUESTIONS:
        query = question["query"]
        target = question["relevant_chunk_id"]

        print("\n" + "=" * 75)
        print(f"Question: {question['id']}")
        print(query)
        print(f"Known relevant chunk: {target}")

        # 1. 原有Hybrid RRF基线
        baseline_start = time.perf_counter()

        hybrid = service.search(
            query=query,
            method="hybrid",
            top_k=FINAL_TOP_K,
            scope="body",
        )

        baseline_time = time.perf_counter() - baseline_start

        # 2. 两路候选召回
        start = time.perf_counter()

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

        retrieval_time = time.perf_counter() - start

        # 3. Cross-Encoder精排
        rerank_start = time.perf_counter()

        reranked = rerank(
            query=query,
            candidates=candidates,
            model=model,
            top_k=FINAL_TOP_K,
        )

        rerank_time = time.perf_counter() - rerank_start

        # 4. 找出目标证据的排名
        baseline_rank = find_rank(
            hybrid["results"], target
        )

        reranker_rank = find_rank(
            reranked, target
        )

        candidate_rank = find_rank(
            candidates, target
        )

        print(
            f"\nDense candidates: {len(dense['results'])}"
        )
        print(
            f"BM25 candidates: {len(bm25['results'])}"
        )
        print(
            f"Union candidates: {len(candidates)}"
        )
        print(
            f"Known evidence in Union: "
            f"{candidate_rank is not None}"
        )

        print_results(
            "Baseline RRF Top-3:",
            hybrid["results"],
        )

        print_results(
            "Cross-Encoder Top-3:",
            reranked,
        )

        print(
            f"\nKnown evidence rank: "
            f"RRF={baseline_rank}, "
            f"Reranker={reranker_rank}"
        )

        print(
            f"Latency: "
            f"baseline={baseline_time:.3f}s, "
            f"retrieval={retrieval_time:.3f}s, "
            f"rerank={rerank_time:.3f}s"
        )

        summary.append({
            "question": question["id"],
            "rrf_hit": baseline_rank is not None,
            "reranker_hit": reranker_rank is not None,
        })

    print("\n" + "=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    for item in summary:
        print(
            f"{item['question']}: "
            f"RRF Hit@3={item['rrf_hit']} | "
            f"Reranker Hit@3={item['reranker_hit']}"
        )

    rrf_hits = sum(
        item["rrf_hit"] for item in summary
    )

    reranker_hits = sum(
        item["reranker_hit"] for item in summary
    )

    print(
        f"\nRRF Known-Evidence Hit@3: "
        f"{rrf_hits}/{len(summary)}"
    )

    print(
        f"Reranker Known-Evidence Hit@3: "
        f"{reranker_hits}/{len(summary)}"
    )


if __name__ == "__main__":
    main()

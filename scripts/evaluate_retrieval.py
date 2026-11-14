
import json
from pathlib import Path

import numpy as np

from rag.embedding_service import EmbeddingService

from scripts.test_docling_retrieval import (
    load_index,
    select_candidates,
    dense_search,
    bm25_search,
    rrf_fusion,
)


EVAL_FILE = Path(
    # "data/evaluation/dcmagrn_retrieval_dev.json"
    "evaluation/dcmagrn_retrieval_dev.json"

)


def reciprocal_rank(results, relevant_ids, chunks, k=5):
    """计算第一个已标注相关Chunk的倒数排名。"""

    relevant = set(relevant_ids)

    for rank, result in enumerate(results[:k], start=1):
        chunk_id = chunks[result["index"]]["chunk_id"]

        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def hit_at_k(results, relevant_ids, chunks, k):
    """前K个结果是否包含至少一个已知相关Chunk。"""

    retrieved = {
        chunks[r["index"]]["chunk_id"]
        for r in results[:k]
    }

    return float(bool(
        retrieved.intersection(relevant_ids)
    ))


def main():
    chunks, vectors, manifest = load_index()

    with EVAL_FILE.open(encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_chunk = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    service = EmbeddingService()

    methods = ["Dense", "BM25", "Hybrid"]
    scores = {
        name: {"Hit@1": [], "Hit@5": [], "RR@5": []}
        for name in methods
    }

    for example in dataset:
        question = example["question"]
        relevant = example["relevant_chunk_ids"]
        scope = example["scope"]

        if not relevant:
            raise ValueError(
                f"{example['query_id']} 未提供已知证据"
            )

        for chunk_id in relevant:
            if chunk_id not in id_to_chunk:
                raise ValueError(
                    f"标准证据Chunk不存在: {chunk_id}"
                )

        candidates = select_candidates(chunks, scope)

        candidate_ids = {
            chunks[i]["chunk_id"]
            for i in candidates
        }

        if not set(relevant).issubset(candidate_ids):
            raise ValueError(
                f"{example['query_id']} 的标准证据"
                f"被 scope={scope} 排除"
            )

        pool_size = min(len(candidates), 20)

        dense = dense_search(
            question, service, vectors,
            candidates, pool_size
        )

        bm25 = bm25_search(
            question, chunks,
            candidates, pool_size
        )

        hybrid = rrf_fusion(
            dense, bm25, top_k=5
        )

        results = {
            "Dense": dense,
            "BM25": bm25,
            "Hybrid": hybrid,
        }

        print(f"\n{example['query_id']}: {question}")

        for name, ranked in results.items():
            h1 = hit_at_k(
                ranked, relevant, chunks, 1
            )
            h5 = hit_at_k(
                ranked, relevant, chunks, 5
            )
            rr5 = reciprocal_rank(
                ranked, relevant, chunks, 5
            )

            scores[name]["Hit@1"].append(h1)
            scores[name]["Hit@5"].append(h5)
            scores[name]["RR@5"].append(rr5)

            top_ids = [
                chunks[item["index"]]["chunk_id"]
                for item in ranked[:5]
            ]

            print(
                f"  {name:7s} "
                f"Hit@1={h1:.0f} "
                f"Hit@5={h5:.0f} "
                f"RR@5={rr5:.3f} "
                f"Top5={top_ids}"
            )

    print("\n" + "=" * 60)
    print("开发集评估汇总")
    print("=" * 60)

    for name in methods:
        metrics = scores[name]

        print(f"\n{name}")

        for key, values in metrics.items():
            print(f"  {key}: {np.mean(values):.4f}")

        print(
            f"  MRR@5: "
            f"{np.mean(metrics['RR@5']):.4f}"
        )


if __name__ == "__main__":
    main()

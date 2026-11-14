RRF_K = 60

#融合 Dense 和 BM25 的排名
def rrf_fusion(dense_results, bm25_results, top_k):
    """融合排名，而不是直接相加原始分数。"""
    fused = {}

    for results in (dense_results, bm25_results):
        for rank, item in enumerate(results, start=1):
            idx = item["index"]

            fused[idx] = (
                fused.get(idx, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    ranked = sorted(
        fused.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [
        {"index": idx, "score": score}
        for idx, score in ranked[:top_k]
    ]

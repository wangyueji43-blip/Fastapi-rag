
from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """
    使用RRF融合多个检索结果。

    每条结果必须包含唯一chunk_id和chunk。
    """

    if top_k < 1 or rrf_k <= 0:
        raise ValueError("top_k和rrf_k必须为正数")

    scores = defaultdict(float)
    documents = {}

    for results in result_lists:
        seen = set()

        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]

            # 防止同一检索器中的重复ID被重复计分
            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            scores[chunk_id] += 1.0 / (rrf_k + rank)
            documents[chunk_id] = item["chunk"]

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            str(chunk_id)
        ),
    )

    return [
        {
            "chunk_id": chunk_id,
            "score": scores[chunk_id],
            "chunk": documents[chunk_id],
        }
        for chunk_id in ranked_ids[:top_k]
    ]

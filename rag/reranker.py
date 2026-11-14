
from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker:
    def __init__(self):
        print("正在加载 Cross-Encoder...")

        self.model = CrossEncoder(
            MODEL_NAME,
            device="cpu",
        )

        print("Cross-Encoder 加载完成")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        对候选 Chunk 进行相关性精排。

        candidates:
            已经召回并格式化的检索结果

        top_k:
            最终保留的结果数量
        """
        if not candidates:
            return []

        # 同时向模型输入 Query 和 Chunk
        pairs = [
            (query, item["content"])
            for item in candidates
        ]

        scores = self.model.predict(
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

        # Cross-Encoder 分数越高，排得越靠前
        ranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # 精排完成以后才截断到 Top-K
        results = ranked[:top_k]

        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        return results


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """
    在当前 Python 进程中缓存一个 Reranker 实例。
    """
    return Reranker()

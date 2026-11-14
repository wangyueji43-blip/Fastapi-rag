
import json
from pathlib import Path

import numpy as np

from rag.embedding_service import EmbeddingService
from rag.retrievers import DenseRetriever, BM25Retriever
from rag.fusion import reciprocal_rank_fusion


DATA_DIR = Path("data/processed")

with (DATA_DIR / "chunks.json").open(
    encoding="utf-8"
) as f:
    chunks = json.load(f)

vectors = np.load(DATA_DIR / "embeddings.npy")

service = EmbeddingService()

dense = DenseRetriever(
    chunks,
    vectors,
    service
)

bm25 = BM25Retriever(chunks)


def show_results(name, results):
    print(f"\n===== {name} =====")

    if not results:
        print("没有检索到候选结果")
        return

    for rank, result in enumerate(results, 1):
        chunk = result["chunk"]

        print(
            f"\n排名: {rank}"
            f" | Chunk ID: {result['chunk_id']}"
            f" | 分数: {result['score']:.4f}"
            f" | 页码: {chunk['page']}"
        )

        print(
            chunk["text"][:250]
            .replace("\n", " ")
        )


questions = [
    "What are the four main modules of DCMAGRN?",
    "Which datasets were used in DCMAGRN?",
    "What is the purpose of the multi-adjacency attention mechanism?",
    "What is GSE178341?",
]

for question in questions:
    print("\n" + "#" * 70)
    print("问题:", question)

    dense_results = dense.search(question, top_k=20)
    bm25_results = bm25.search(question, top_k=20)

    hybrid_results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        top_k=5,
    )

    show_results("Dense Top-5", dense_results[:5])
    show_results("BM25 Top-5", bm25_results[:5])
    show_results("Hybrid Top-5", hybrid_results)

import argparse

from rag.embedding_service import EmbeddingService
from rag.index_store import load_index
from rag.docling_retrieval import (
    select_candidates,
    dense_search,
    bm25_search,
)
from rag.docling_fusion import rrf_fusion


# def tokenize(text):
#     """英文科研文本的基础BM25分词器。"""
#     return re.findall(
#         r"[a-zA-Z0-9]+",
#         text.lower(),
#     )


# def sha256_file(path):
#     digest = hashlib.sha256()

#     with path.open("rb") as f:
#         for block in iter(
#             lambda: f.read(1024 * 1024), b""
#         ):
#             digest.update(block)

#     return digest.hexdigest()


# def load_index():
#     with (INDEX_DIR / "chunks.json").open(
#         encoding="utf-8"
#     ) as f:
#         chunks = json.load(f)

#     with (INDEX_DIR / "manifest.json").open(
#         encoding="utf-8"
#     ) as f:
#         manifest = json.load(f)

#     vectors = np.load(
#         INDEX_DIR / "embeddings.npy",
#         allow_pickle=False,
#     )

#     # 校验索引完整性
#     assert vectors.ndim == 2
#     assert len(chunks) == len(vectors)
#     assert len(chunks) == manifest["chunk_count"]
#     assert vectors.shape[1] == manifest["embedding_dim"]
#     assert manifest["normalized"] is True
#     assert manifest["text_field"] == "embed_text"

#     ids = [c["chunk_id"] for c in chunks]

#     if len(set(ids)) != len(ids):
#         raise ValueError("存在重复Chunk ID")

#     if ids != manifest["row_to_chunk_id"]:
#         raise ValueError("Chunk与Embedding顺序不一致")

#     if not np.isfinite(vectors).all():
#         raise ValueError("Embedding包含非有限数值")

#     norms = np.linalg.norm(vectors, axis=1)

#     if not np.allclose(norms, 1.0, atol=1e-4):
#         raise ValueError("Embedding没有正确归一化")

#     source = Path(manifest["source_file"])

#     if source.exists():
#         if sha256_file(source) != manifest["source_sha256"]:
#             raise ValueError("源Chunk文件已发生变化")

#     return chunks, vectors, manifest


# def select_candidates(chunks, scope):
#     if scope == "all":
#         return list(range(len(chunks)))

#     return [
#         i for i, chunk in enumerate(chunks)
#         if chunk["content_type"] == scope
#     ]


# def dense_search(
#     question, service, vectors, candidates, top_k
# ):
#     query_vector = np.asarray(
#         service.encode_query(question),
#         dtype=np.float32,
#     ).reshape(-1)

#     if query_vector.shape[0] != vectors.shape[1]:
#         raise ValueError("Query与Document向量维度不一致")

#     norm = np.linalg.norm(query_vector)

#     if not np.isfinite(norm) or norm < 1e-8:
#         raise ValueError("Query Embedding异常")

#     query_vector /= norm

#     candidate_vectors = vectors[candidates]
#     scores = candidate_vectors @ query_vector

#     order = np.argsort(-scores, kind="stable")[:top_k]

#     return [
#         {
#             "index": candidates[int(i)],
#             "score": float(scores[i]),
#         }
#         for i in order
#     ]


# def bm25_search(question, chunks, candidates, top_k):
#     corpus = [
#         tokenize(chunks[i]["embed_text"])
#         for i in candidates
#     ]

#     if not any(corpus):
#         return []

#     bm25 = BM25Okapi(corpus)

#     query_tokens = tokenize(question)

#     if not query_tokens:
#         return []

#     scores = bm25.get_scores(query_tokens)
#     order = np.argsort(-scores, kind="stable")[:top_k]

#     return [
#         {
#             "index": candidates[int(i)],
#             "score": float(scores[i]),
#         }
#         for i in order
#     ]


# def rrf_fusion(dense_results, bm25_results, top_k):
#     """融合排名，而不是直接相加原始分数。"""
#     fused = {}

#     for results in (dense_results, bm25_results):
#         for rank, item in enumerate(results, start=1):
#             idx = item["index"]

#             fused[idx] = (
#                 fused.get(idx, 0.0)
#                 + 1.0 / (RRF_K + rank)
#             )

#     ranked = sorted(
#         fused.items(),
#         key=lambda item: (-item[1], item[0]),
#     )

#     return [
#         {"index": idx, "score": score}
#         for idx, score in ranked[:top_k]
#     ]


def show_results(name, results, chunks):
    print(f"\n{'=' * 65}")
    print(name)
    print("=" * 65)

    for rank, result in enumerate(results, start=1):
        chunk = chunks[result["index"]]

        print(f"\nRank {rank}")
        print("ID:", chunk["chunk_id"])
        print("Score:", round(result["score"], 5))
        print("Pages:", chunk["pages"])
        print("Headings:", chunk["headings"])
        print("Type:", chunk["content_type"])
        print(
            "Text:",
            chunk["text"][:350].replace("\n", " "),
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--scope",
        choices=["body", "reference", "all"],
        default="body",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k 必须大于0")

    chunks, vectors, manifest = load_index()

    print("Index模型:", manifest["model_name"])
    print("Chunk数量:", len(chunks))
    print("向量Shape:", vectors.shape)

    candidates = select_candidates(chunks, args.scope)

    if not candidates:
        raise ValueError("当前检索范围没有候选Chunk")

    print("检索范围:", args.scope)
    print("候选数量:", len(candidates))

    service = EmbeddingService()

    # RRF使用相对更大的候选池
    pool_size = min(
        len(candidates),
        max(args.top_k * 4, 20),
    )

    dense = dense_search(
        args.query,
        service,
        vectors,
        candidates,
        pool_size,
    )

    bm25 = bm25_search(
        args.query,
        chunks,
        candidates,
        pool_size,
    )

    hybrid = rrf_fusion(
        dense,
        bm25,
        args.top_k,
    )

    show_results(
        "Dense Top-K", dense[:args.top_k], chunks
    )

    show_results(
        "BM25 Top-K", bm25[:args.top_k], chunks
    )

    show_results(
        "Hybrid RRF Top-K", hybrid, chunks
    )


if __name__ == "__main__":
    main()

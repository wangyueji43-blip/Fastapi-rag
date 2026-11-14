import re

import numpy as np
from rank_bm25 import BM25Okapi

#将文本拆成 BM25 使用的词项 Token
def tokenize(text):
    """英文科研文本的基础BM25分词器。"""
    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

#根据 scope 筛选参与检索的 Chunk
def select_candidates(chunks, scope):
    if scope == "all":
        return list(range(len(chunks)))

    return [
        i for i, chunk in enumerate(chunks)
        if chunk["content_type"] == scope
    ]

#计算查询向量与文档向量的相似度并排序
def dense_search(
    question, service, vectors, candidates, top_k
):
    query_vector = np.asarray(
        service.encode_query(question),
        dtype=np.float32,
    ).reshape(-1)

    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError("Query与Document向量维度不一致")

    norm = np.linalg.norm(query_vector)

    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Query Embedding异常")

    query_vector /= norm

    candidate_vectors = vectors[candidates]
    scores = candidate_vectors @ query_vector

    order = np.argsort(-scores, kind="stable")[:top_k]

    return [
        {
            "index": candidates[int(i)],
            "score": float(scores[i]),
        }
        for i in order
    ]

#根据词项匹配和 BM25 相关性得分排序
def bm25_search(question, chunks, candidates, top_k):
    corpus = [
        tokenize(chunks[i]["embed_text"])
        for i in candidates
    ]

    if not any(corpus):
        return []

    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    order = np.argsort(-scores, kind="stable")[:top_k]

    return [
        {
            "index": candidates[int(i)],
            "score": float(scores[i]),
        }
        for i in order
    ]
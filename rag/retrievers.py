
import re
import numpy as np
import jieba

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    基础中英文分词。
    英文、数字、技术标识符使用正则提取；
    连续中文字符使用 jieba 分词。
    """
    text = text.lower()

    segments = re.findall(
        r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u4e00-\u9fff]+",
        text
    )

    tokens = []

    for segment in segments:
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(
                word for word in jieba.cut(segment)
                if word.strip()
            )
        else:
            tokens.append(segment)

    return tokens


class DenseRetriever:
    """基于归一化向量点积的语义检索器。"""

    def __init__(self, chunks, vectors, embedding_service):
        self.chunks = chunks
        self.vectors = np.asarray(
            vectors,
            dtype=np.float32
        )
        self.embedding_service = embedding_service

        if len(chunks) != len(self.vectors):
            raise ValueError("Chunk与向量数量不一致")

        if self.vectors.ndim != 2:
            raise ValueError("向量矩阵必须是二维数组")

    def search(self, query: str, top_k: int = 10):
        if top_k < 1:
            raise ValueError("top_k必须大于0")

        query_vector = self.embedding_service.encode_query(query)
        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        if query_vector.shape != (self.vectors.shape[1],):
            raise ValueError("查询向量维度与索引不匹配")

        scores = self.vectors @ query_vector
        indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            {
                "index": int(idx),
                "chunk_id": self.chunks[int(idx)]["chunk_id"],
                "score": float(scores[idx]),
                "chunk": self.chunks[int(idx)],
            }
            for idx in indices
        ]


class BM25Retriever:
    """基于BM25的词项检索器。"""

    def __init__(self, chunks):
        self.chunks = chunks

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        # 防止空文本或空词表导致无效索引
        if not self.tokenized_corpus or not any(
            self.tokenized_corpus
        ):
            raise ValueError("BM25语料库不能为空")

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10):
        if top_k < 1:
            raise ValueError("top_k必须大于0")

        tokens = tokenize(query)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # 仅返回具有正BM25得分的候选，
        # 避免把无匹配的Chunk当成有效结果。
        indices = [
            int(idx)
            for idx in np.argsort(-scores, kind="stable")
            if np.isfinite(scores[idx]) and scores[idx] > 0
        ][:top_k]

        return [
            {
                "index": idx,
                "chunk_id": self.chunks[idx]["chunk_id"],
                "score": float(scores[idx]),
                "chunk": self.chunks[idx],
            }
            for idx in indices
        ]

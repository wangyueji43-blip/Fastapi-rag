#使用本地模型sencetence transformer生成embedding
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingService:

    def __init__(self):
        print("正在加载 Embedding 模型...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        print("Embedding 模型加载完成")

    def encode_documents(
        self,
        texts: list[str]
    ) -> np.ndarray:
        """批量向量化文档片段。"""

        vectors = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vectors

    def encode_query(
        self,
        question: str
    ) -> np.ndarray:
        """向量化用户问题。"""

        vector = self.model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vector
"""
这里有三个重要知识点
① 为什么把模型加载放进 __init__()？
self.model = SentenceTransformer(...)


这是创建类实例时执行的初始化逻辑。

以后只需要：
embedding_service = EmbeddingService()


便能够复用同一个模型，避免每处理一个 Chunk 就重新加载一次模型。

② batch_size=16 有什么用？

它表示模型一次处理最多 16 个文本输入。

假设我们有 160 个 Chunk，就可以分批编码，而不是一个个调用模型。

Batching（批处理）通常有利于提升推理效率，但实际效果受 CPU、GPU、内存和输入长度等因素影响。

③ normalize_embeddings=True 为什么重要？

它会对输出向量进行 L2 归一化，使其长度约等于 1。

对于两个归一化后的向量：
\[
\|A\|=\|B\|=1
\]

因此：
\[
\operatorname{cosine}(A,B)=A\cdot B
\]

也就是说，可以直接用点积计算余弦相似度，无需每次重复计算范数。这也是 Sentence Transformers 文档说明的优化方式。
Sentence Transformers documentation
"""
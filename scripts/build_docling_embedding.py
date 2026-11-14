
import hashlib
import json
from pathlib import Path

import numpy as np

from rag.embedding_service import EmbeddingService


INPUT_PATH = Path(
    "data/processed/docling/chunks_tagged.json"
)

OUTPUT_DIR = Path(
    "data/processed/docling/index"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

BATCH_SIZE = 32


def file_sha256(path):
    """计算文件指纹，用于追踪数据版本。"""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    # 1. 加载经过分类的Chunk
    with INPUT_PATH.open(encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("Chunk数据为空")

    # 2. 检查Chunk ID及内容
    ids = [c["chunk_id"] for c in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("存在重复chunk_id")

    valid_types = {"body", "reference"}

    for chunk in chunks:
        if not chunk.get("embed_text", "").strip():
            raise ValueError(
                f"空Embedding文本: {chunk['chunk_id']}"
            )

        if chunk.get("content_type") not in valid_types:
            raise ValueError(
                f"内容类型不合法: {chunk['chunk_id']}"
            )

    # 3. 保留明确的数据顺序
    texts = [
        chunk["embed_text"]
        for chunk in chunks
    ]

    # 4. 复用原来的EmbeddingService
    service = EmbeddingService()

    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        batch_vectors = service.encode_documents(batch)

        batch_vectors = np.asarray(
            batch_vectors,
            dtype=np.float32,
        )

        if batch_vectors.ndim != 2:
            raise ValueError(
                "Embedding结果不是二维矩阵"
            )

        vectors.append(batch_vectors)

        print(
            f"已编码: {min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}"
        )

    embeddings = np.vstack(vectors)

    # 5. 检查维度和数据完整性
    if embeddings.shape[0] != len(chunks):
        raise ValueError("向量数量与Chunk数量不匹配")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding包含NaN或Inf")

    # 6. 检查向量范数
    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms < 1e-8):
        raise ValueError("存在零向量或异常小范数向量")

    # 统一L2归一化，便于后续点积检索
    embeddings = embeddings / norms[:, None]

    # 7. 构建索引元数据
    manifest = {
        "model_name": MODEL_NAME,
        "chunk_count": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "text_field": "embed_text",
        "source_file": str(INPUT_PATH),
        "source_sha256": file_sha256(INPUT_PATH),
        "row_to_chunk_id": ids,
    }

    # 8. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        OUTPUT_DIR / "embeddings.npy",
        embeddings,
    )

    (OUTPUT_DIR / "chunks.json").write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== Embedding构建完成 ==========")
    print("Chunk数量:", len(chunks))
    print("Embedding Shape:", embeddings.shape)
    print("数据类型:", embeddings.dtype)
    print("向量范数最小值:", round(float(norms.min()), 5))
    print("向量范数最大值:", round(float(norms.max()), 5))
    print("输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

import hashlib
import json
from pathlib import Path
import numpy as np

INDEX_DIR = Path("data/processed/docling/index")

#计算文件 SHA-256 哈希，用来检测文件是否发生变化
def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()

#加载并校验chunk、embedding和manifest
def load_index():
    with (INDEX_DIR / "chunks.json").open(
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    with (INDEX_DIR / "manifest.json").open(
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    vectors = np.load(
        INDEX_DIR / "embeddings.npy",
        allow_pickle=False,
    )

    # 校验索引完整性
    assert vectors.ndim == 2
    assert len(chunks) == len(vectors)
    assert len(chunks) == manifest["chunk_count"]
    assert vectors.shape[1] == manifest["embedding_dim"]
    assert manifest["normalized"] is True
    assert manifest["text_field"] == "embed_text"

    ids = [c["chunk_id"] for c in chunks]

    if len(set(ids)) != len(ids):
        raise ValueError("存在重复Chunk ID")

    if ids != manifest["row_to_chunk_id"]:
        raise ValueError("Chunk与Embedding顺序不一致")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding包含非有限数值")

    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embedding没有正确归一化")

    source = Path(manifest["source_file"])

    if source.exists():
        if sha256_file(source) != manifest["source_sha256"]:
            raise ValueError("源Chunk文件已发生变化")

    return chunks, vectors, manifest


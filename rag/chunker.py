#chunking函数
"""
chunk_size=400：每个 Chunk 最多取 400 个字符。
chunk_overlap=80：相邻 Chunk 之间重复 80 个字符。
"""
# chunking的权衡
"""
Chunk 太小：容易割裂语义
Chunk 太大：检索定位不够精确，还可能引入无关内容。
Overlap 太小：边界附近的上下文可能丢失。
Overlap 太大：存储和检索冗余增加。
"""

def split_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    按字符长度对每页文本进行滑动窗口切分。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap 必须大于等于0，且小于chunk_size"
        )

    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(
                start + chunk_size,
                len(text)
            )

            content = text[start:end].strip()

            if content:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": page["source"],
                    "page": page["page"],
                    "text": content,
                })

            # 已到达页面结尾，避免重复
            if end == len(text):
                break

            # 让下一个 Chunk 保留一部分上下文
            start = end - chunk_overlap

    return chunks

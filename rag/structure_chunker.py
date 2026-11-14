#第二步：实现结构感知 Chunking
"""
这是今天的核心算法。
我们的处理过程分为四个操作：
1
识别章节标题
例如 Introduction、Dataset、Framework、Discussion。
2
按段落提取文本单元
优先利用 PDF 提取后保留的空行，不在段落中间随意截断。
3
长段落按句子继续拆分
若一个段落超过长度限制，再沿句号、问号等句子边界拆分。
4
组合文本单元
使用长度预算将完整单元组合成 Chunk，并尽量保留少量重叠上下文。


注意，我们需要区分两个概念：
- 文本单元（Unit）：一个完整段落、句子或被迫截断的长句片段。
- Chunk：由一个或多个文本单元组装而成、最终用于 Embedding 和检索的内容。
1. 为什么不能简单地按段落切完就结束？
例如：
Paragraph A: 200 chars
Paragraph B: 150 chars
Paragraph C: 300 chars


假设 chunk_size=500。

我们希望：
Chunk 1: A + B = 350 chars
Chunk 2: C = 300 chars


但如果 Paragraph C 有 900 个字符，就不能直接作为一个完整 Chunk。

我们就需要递归降低分割粒度：
段落
 ↓ 超过长度
句子
 ↓ 仍超过长度
词边界
 ↓ 单个连续词仍过长
最终强制拆分

这就是 Recursive Chunking 的核心思想：优先使用高层级语义边界，在必要时才使用更细粒度边界。"""

import re
import textwrap

from rag.text_cleaner import clean_pdf_text


SECTION_NAMES = {
    "abstract",
    "introduction",
    "related work",
    "materials and methods",
    "methods",
    "dataset",
    "framework",
    "gene embedding module",
    "feature updating module",
    "channel updating module",
    "model training",
    "results",
    "ablation study",
    "discussion",
    "conclusion",
    "references",
}


def is_heading(text: str) -> bool:
    """判断是否为已知的独立章节标题。"""
    normalized = re.sub(
        r"\s+", " ", text
    ).strip().lower()

    return normalized in SECTION_NAMES


def split_long_text(
    text: str,
    max_chars: int,
) -> list[str]:
    """
    将超长段落优先按句子拆分，
    若句子仍过长，再按词边界拆分。
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?。！？])\s+",
        text
    )

    units = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) <= max_chars:
            units.append(sentence)
            continue

        # 长句按单词边界尽量拆开
        parts = textwrap.wrap(
            sentence,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        units.extend(parts)

    return units


# def pack_units(
#     units: list[str],
#     chunk_size: int,
#     chunk_overlap: int,
# ) -> list[str]:
#     """
#     将文本单元组装成Chunk。
#     尽可能保留完整单元与重叠上下文。
#     """
#     chunks = []
#     current = []

#     for unit in units:
#         unit = unit.strip()

#         if not unit:
#             continue

#         separator = "\n\n" if current else ""

#         current_text = separator.join(current)

#         additional = (
#             len(unit)
#             + (2 if current else 0)
#         )

#         if (
#             current
#             and len(current_text) + additional > chunk_size
#         ):
#             chunks.append(current_text)

#             # 为下一个Chunk选择可容纳的尾部单元
#             overlap_units = []
#             limit = min(
#                 chunk_overlap,
#                 max(0, chunk_size - len(unit) - 2)
#             )

#             for previous in reversed(current):
#                 candidate = [previous] + overlap_units
#                 candidate_text = "\n\n".join(candidate)

#                 if len(candidate_text) > limit:
#                     break

#                 overlap_units = candidate

#             current = overlap_units

#         current.append(unit)

#     if current:
#         final_text = "\n\n".join(current)

#         if not chunks or final_text != chunks[-1]:
#             chunks.append(final_text)

#     return chunks

def pack_units(
    units: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    将文本单元按原顺序组装为Chunk。

    约束：
    1. Chunk长度不超过chunk_size
    2. 尽可能保留完整Unit
    3. overlap不超过chunk_overlap
    4. 不遗漏任何非空Unit
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap参数不合法")

    separator = "\n\n"

    # 空Unit不作为检索内容
    normalized_units = [
        unit.strip()
        for unit in units
        if unit.strip()
    ]

    # pack_units只负责组装，不负责拆分超长Unit
    for unit in normalized_units:
        if len(unit) > chunk_size:
            raise ValueError(
                f"Unit长度{len(unit)}超过"
                f"chunk_size={chunk_size}"
            )

    chunks = []
    current = []

    for unit in normalized_units:
        candidate = separator.join(current + [unit])

        # 当前Unit可以直接加入
        if len(candidate) <= chunk_size:
            current.append(unit)
            continue

        # 当前Chunk已经装不下新Unit
        chunks.append(separator.join(current))

        # 从旧Chunk尾部选取完整Unit作为Overlap
        overlap_units = []

        for previous in reversed(current):
            proposed = [previous] + overlap_units
            overlap_text = separator.join(proposed)

            if len(overlap_text) > chunk_overlap:
                break

            # 重叠文本 + 新Unit也必须装得下
            next_text = separator.join(proposed + [unit])

            if len(next_text) > chunk_size:
                break

            overlap_units = proposed

        current = overlap_units + [unit]

    if current:
        chunks.append(separator.join(current))

    # 最终校验长度不变量
    assert all(
        len(chunk) <= chunk_size
        for chunk in chunks
    )

    return chunks



def structure_chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    基于章节、段落和句子边界切分PDF页面。
    保留source、page、section等元数据。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap参数不合法")

    chunks = []
    current_section = "Unknown"

    for page in pages:
        text = clean_pdf_text(page["text"])

        # 优先按空白行识别段落
        paragraphs = re.split(r"\n\s*\n", text)

        current_units = []

        def flush_section():
            nonlocal current_units

            packed = pack_units(
                current_units,
                chunk_size,
                chunk_overlap,
            )

            for content in packed:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": page["source"],
                    "page": page["page"],
                    "section": current_section,
                    "text": content,
                })

            current_units = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # 判断整个段落是否为标题
            if is_heading(paragraph):
                flush_section()
                current_section = paragraph
                continue

            # 合并普通段落中的物理换行
            paragraph = re.sub(
                r"\s*\n\s*",
                " ",
                paragraph
            )

            # 超长段落进一步拆分
            units = split_long_text(
                paragraph,
                chunk_size
            )

            current_units.extend(units)

        flush_section()

    return chunks

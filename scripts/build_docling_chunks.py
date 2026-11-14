
import json
from pathlib import Path
from statistics import mean, median

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer


# 与现有Embedding模型保持一致
EMBED_MODEL_ID = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# MODEL_MAX_TOKENS = 128
# CHUNK_MAX_TOKENS = 120

INPUT_PATH = Path(
    "data/processed/docling/sample.json"
)

OUTPUT_PATH = Path(
    "data/processed/docling/chunks.json"
)


def main():
    # 1. 读取已经解析好的结构化PDF
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    doc = DoclingDocument.model_validate_json(
        INPUT_PATH.read_text(encoding="utf-8")
    )
    """
    之前 sample.json 是一个保存在硬盘上的 JSON 文件。
    这一步使用 Pydantic 的数据模型校验和反序列化功能，将它恢复成可操作的 DoclingDocument 对象。
    你可以把它理解为：
    硬盘上的 JSON
        ↓
    解析与校验
        ↓
    Python 中的 DoclingDocument 对象
        ↓
    可以调用 HybridChunker
    这与直接读取 Markdown 不一样。Markdown 虽然容易阅读，但未必保留完整的文档结构关系和页面来源信息。
    """
    # 2. 加载Embedding模型对应的Tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(
        EMBED_MODEL_ID,
        use_fast=True,
    )

    # tokenizer = HuggingFaceTokenizer(
    #     tokenizer=hf_tokenizer,
    #     max_tokens=CHUNK_MAX_TOKENS,
    # )
    # 模型实际输入的最大长度
    MODEL_MAX_TOKENS = 128

    # 我们希望最终Chunk最多占用的Token数
    CHUNK_MAX_TOKENS = 120

    assert CHUNK_MAX_TOKENS <= MODEL_MAX_TOKENS
    
    # 自动计算特殊Token开销
    special_tokens = hf_tokenizer.num_special_tokens_to_add(
        pair=False
    )

    # Docling内部不计特殊Token，因此提前扣除
    docling_budget = CHUNK_MAX_TOKENS - special_tokens

    if docling_budget <= 0:
        raise ValueError("Chunk预算不足")

    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=docling_budget,
    )

    print("模型最大Token:", MODEL_MAX_TOKENS)
    print("Chunk目标上限:", CHUNK_MAX_TOKENS)
    print("特殊Token:", special_tokens)
    print("Docling正文预算:", docling_budget)
    """
    这里涉及两层接口：
    - AutoTokenizer：Hugging Face 提供的真实分词器，负责把文本转换成 Token ID。
    - HuggingFaceTokenizer：Docling 提供的适配器，让 HybridChunker 可以用统一接口调用分词器并检查 Token 数。
    这就是一种典型的 Adapter Pattern（适配器模式）。
    它并没有重新训练 Tokenizer，只是让不同软件组件之间的接口能够对接。
    """
    # 3. 初始化Docling HybridChunker
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    # 4. 执行结构化Chunking
    chunks = list(chunker.chunk(dl_doc=doc))

    records = []

    for i, chunk in enumerate(chunks):
        # 原始正文，不附加章节标题
        raw_text = chunk.text

        # 用于Embedding的文本：包含章节上下文
        embed_text = chunker.contextualize(chunk)
        """
        chunk.text 和 chunker.contextualize(chunk) 为什么不一样？
        这是这次最值得理解的地方。
        例如，假设某个 Chunk 只有一句正文：
        The model integrates four adjacency matrices.
        那么：
        chunk.text
        可能就是这段正文。
        但是：
        chunker.contextualize(chunk)
        可能变成：
        DCMAGRN
        Feature Updating Module
        The model integrates four adjacency matrices.
        也就是把章节上下文一起加入。
        对于 RAG 来说，这通常很有意义。
        因为如果只检索：
        The model integrates four adjacency matrices.
        模型可能不清楚这句话属于哪个方法模块。
        而增加章节上下文后，Embedding 输入可以包含更明确的语义信息。
        Docling 官方也建议在检索场景中关注 contextualize() 生成的上下文增强文本。
        Docling
        但需要注意，增加标题不保证一定提升检索效果，最终仍然要通过测试集验证。
        """
        # 使用真实Tokenizer重新统计Token
        token_ids = hf_tokenizer(
            embed_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        token_count = len(token_ids)
        
        # 从来源文档元素中收集页码
        pages = sorted({
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov
        })

        headings = list(chunk.meta.headings or [])

        records.append({
            "chunk_id": f"docling_{i:04d}",
            "source": "sample.pdf",
            "pages": pages,
            "headings": headings,
            "text": raw_text,
            "embed_text": embed_text,
            "token_count": token_count,
            "char_count": len(embed_text),
        })

    # 5. 质量检查
    if not records:
        raise RuntimeError("HybridChunker未生成Chunk")

    if any(not r["embed_text"].strip() for r in records):
        raise RuntimeError("存在空Chunk")

    print(
    "特殊Token数量:",
    hf_tokenizer.num_special_tokens_to_add(
        pair=False
    )
    )

    for chunk in chunks[:5]:
        text = chunker.contextualize(chunk)

        docling_count = tokenizer.count_tokens(text)

        actual_count = len(
            hf_tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )

        print(
            "Docling计数:", docling_count,
            "模型输入计数:", actual_count,
            "差值:", actual_count - docling_count,
        )

    oversized = [
        r for r in records
        if r["token_count"] > CHUNK_MAX_TOKENS
    ]

    if oversized:
        print("\n发现超过Chunk预算的内容:")
        for r in oversized[:10]:
            print(r["chunk_id"], r["token_count"])

        raise RuntimeError(
            "存在超出Token预算的Chunk，"
            "需要检查切分或序列化策略"
        )

    # 6. 保存
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lengths = [r["token_count"] for r in records]

    print("\n========== Chunking结果 ==========")
    print("Chunk数量:", len(records))
    print("最短Token数:", min(lengths))
    print("最长Token数:", max(lengths))
    print("平均Token数:", round(mean(lengths), 1))
    print("中位Token数:", median(lengths))
    print("超过预算:", len(oversized))
    print("超过模型限制:", sum(
        n > MODEL_MAX_TOKENS for n in lengths
    ))

    print("\n========== 前5个Chunk ==========")

    for r in records[:5]:
        print("\n" + "=" * 50)
        print("ID:", r["chunk_id"])
        print("Pages:", r["pages"])
        print("Headings:", r["headings"])
        print("Tokens:", r["token_count"])
        print("Text:", r["embed_text"][:450])

    print("\n保存位置:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

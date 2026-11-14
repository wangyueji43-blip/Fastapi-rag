
import re
from pathlib import Path

import pymupdf


def extract_text_blocks(page):
    """提取页面文本块，保留原始坐标与块编号。"""
    result = []

    for block in page.get_text("blocks", sort=False):
        if block[6] != 0:
            continue

        text = block[4].strip()

        if not text:
            continue

        result.append({
            "block_id": int(block[5]),
            "bbox": [float(v) for v in block[:4]],
            "text": text,
        })

    return result


def is_template_noise(block, page_height):
    """
    保守过滤当前论文中已确认的模板信息。
    不按位置直接删除全部顶部或底部内容。
    """
    text = " ".join(block["text"].split())
    y0 = block["bbox"][1]

    # 顶部的期刊模板标记
    if y0 < 110:
        patterns = [
            r"^Journal Title Here",
            r"^doi:\s*DOI HERE",
            r"^Paper$",
            r"^\d+\s+Author Name et al\.$",
            """
            ^：从字符串开头匹配。
            \d+：一个或多个数字。
            \s+：一个或多个空白字符。
            Author Name et al\.：作者占位文字。
            $：匹配到字符串结尾。            
            """
        ]

        if any(
            re.search(p, text, flags=re.IGNORECASE)
            for p in patterns
        ):
            return True

    # 当前论文中明确的版权模板文字
    if y0 > page_height - 65:
        if text.startswith("© The Author 2022"):
            return True

        # 底部独立页码
        if re.fullmatch(r"\d{1,3}", text):
            return True

    return False


def classify_block(block, mid_x, page_width):
    """
    根据BBox识别左栏、右栏或跨栏。

    WIDE阈值是初始启发式参数，
    不是通用PDF标准。
    """
    x0, _, x1, _ = block["bbox"]
    width = x1 - x0

    crosses = x0 < mid_x < x1

    # 跨越中线且覆盖较大横向范围
    if crosses and width >= page_width * 0.60:
        return "wide"

    # 其余按文本块中心所属区域划分
    center_x = (x0 + x1) / 2

    if center_x < mid_x:
        return "left"

    return "right"


def order_column_blocks(blocks, mid_x, page_width):
    """分别排序左、右栏，再按阅读顺序合并。"""
    left = []
    right = []

    for block in blocks:
        region = classify_block(
            block, mid_x, page_width
        )

        if region == "left":
            left.append(block)
        else:
            right.append(block)

    left.sort(
        key=lambda b: (b["bbox"][1], b["bbox"][0])
    )
    right.sort(
        key=lambda b: (b["bbox"][1], b["bbox"][0])
    )

    return left + right


def reconstruct_page(page):
    """
    根据跨栏Block切分版面区域，
    在每个区域内按左栏到右栏排序。
    """
    all_blocks = extract_text_blocks(page)

    mid_x = (page.rect.x0 + page.rect.x1) / 2
    page_width = page.rect.width

    kept = []
    filtered = []

    for block in all_blocks:
        if is_template_noise(
            block, page.rect.height
        ):
            filtered.append(block)
        else:
            kept.append(block)

    # 初始按视觉纵向位置排序
    kept.sort(
        key=lambda b: (b["bbox"][1], b["bbox"][0])
    )

    ordered = []
    pending_columns = []
    warnings = []

    def flush_columns():
        nonlocal pending_columns

        ordered.extend(
            order_column_blocks(
                pending_columns,
                mid_x,
                page_width,
            )
        )
        pending_columns = []

    for block in kept:
        region = classify_block(
            block, mid_x, page_width
        )

        if region == "wide":
            wide_top = block["bbox"][1]

            # 若跨栏块与之前的栏内块垂直重叠，
            # 简单的区域切分可能存在阅读顺序歧义。
            overlapping = [
                b for b in pending_columns
                if b["bbox"][3] > wide_top
            ]

            if overlapping:
                warnings.append({
                    "type": "vertical_overlap",
                    "wide_block": block["block_id"],
                    "other_blocks": [
                        b["block_id"] for b in overlapping
                    ],
                })

            flush_columns()
            ordered.append(block)

        else:
            # 跨中线但宽度不足的块，分类有歧义
            x0, _, x1, _ = block["bbox"]

            if x0 < mid_x < x1:
                warnings.append({
                    "type": "ambiguous_midline",
                    "block_id": block["block_id"],
                })

            pending_columns.append(block)

    flush_columns()

    # 检查：重排序不能丢失或复制文本块
    input_ids = [
        b["block_id"] for b in kept
    ]
    output_ids = [
        b["block_id"] for b in ordered
    ]

    if (
        len(input_ids) != len(output_ids)
        or set(input_ids) != set(output_ids)
    ):
        raise RuntimeError(
            "阅读顺序恢复出现Block丢失或重复"
        )

    text = "\n\n".join(
        b["text"] for b in ordered
    )

    return {
        "text": text,
        "blocks": ordered,
        "filtered_blocks": filtered,
        "warnings": warnings,
    }


def load_pdf_layout(pdf_path):
    """解析完整PDF，保留逐页元数据。"""
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    results = []

    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            parsed = reconstruct_page(page)

            results.append({
                "source": path.name,
                "page": index + 1,
                **parsed,
            })

    return results
"""
这份代码的三个重点
第一：classify_block()
通过文本块的横坐标判断其处于左栏、右栏还是跨栏。
这里使用 width >= page_width * 0.60 判断一个跨中线 Block 是否足够宽。它只是当前实验的初始阈值，不是 PyMuPDF 定义的跨栏标准。
第二：flush_columns()
把当前积累的左右栏文本按正确顺序写入最终结果。
它体现了我们上一节学习的思想：先积累内容，遇到边界时再处理。
第三：warnings
如果一个跨栏 Block 与其他文本块在纵向位置上重叠，就记录潜在的阅读顺序歧义。
为什么不直接报错？
因为在多页论文中，公式、图注或表格经常会与其他区域产生复杂的坐标关系。我们先保留全部内容并记录异常，再对问题页面进行人工检查。
注意：这份实现不会裁剪原始文本块，但不代表自动重建的阅读顺序一定正确。 特别是被归类为跨栏的图片文字、公式或表格，仍可能需要单独处理。
"""

from collections import defaultdict

from docling_core.types.doc import DoclingDocument

from collections import defaultdict


def detect_column_switches(doc):
    pages = defaultdict(list)

    for order, (item, level) in enumerate(
        doc.iterate_items()
    ):
        text = getattr(item, "text", "") or ""

        if not text.strip():
            continue

        for prov in getattr(item, "prov", []):
            bbox = prov.bbox

            pages[prov.page_no].append({
                "order": order,
                "text": text[:120],
                "label": str(getattr(item, "label", "")),
                "left": float(bbox.l),
                "right": float(bbox.r),
            })

    reports = []

    for page_no, items in pages.items():
        if len(items) < 4:
            continue

        left_edge = min(i["left"] for i in items)
        right_edge = max(i["right"] for i in items)

        mid_x = (left_edge + right_edge) / 2
        page_width = right_edge - left_edge

        sequence = []

        for item in items:
            left = item["left"]
            right = item["right"]

            center_x = (left + right) / 2
            width = right - left

            # 显著跨越中线的文本块不参与左右栏跳转检测
            if (
                left < mid_x < right
                and width > page_width * 0.65
            ):
                side = "WIDE"
            elif center_x < mid_x:
                side = "L"
            else:
                side = "R"

            sequence.append({
                **item,
                "side": side,
            })

        # 仅对明确位于左右栏的文本进行切换检查
        column_items = [
            item for item in sequence
            if item["side"] in {"L", "R"}
        ]

        switches = []

        for i in range(1, len(column_items)):
            previous = column_items[i - 1]
            current = column_items[i]

            if previous["side"] != current["side"]:
                switches.append({
                    "from": previous["side"],
                    "to": current["side"],
                    "order": current["order"],
                    "text": current["text"],
                })

        # 多次左右切换，记录为异常候选
        if len(switches) > 1:
            reports.append({
                "page": page_no,
                "issue": "suspected_column_interleaving",
                "switch_count": len(switches),
                "switches": switches,
            })

    return reports

"""
第一，order 是 Docling 文档元素的逻辑遍历顺序。
我们要检查的就是这个顺序是否符合页面布局，
而不是重新根据 y 坐标排序。
第二，当前中线是根据文本边界估计出来的。
这是一个用于双栏文档异常检测的初始方法。
它不能可靠处理三栏、非对称栏宽或复杂跨栏布局。
后面可以结合版面检测结果来决定某页是否适合运行这项检查。
"""
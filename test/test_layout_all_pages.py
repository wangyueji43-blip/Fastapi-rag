
import json
from pathlib import Path

from rag.layout_pdf_loader import load_pdf_layout


PDF_PATH = "data/papers/sample.pdf"
OUTPUT_DIR = Path("data/processed/layout_v2")


def main():
    pages = load_pdf_layout(PDF_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 保存结构化JSON，包含Block坐标与异常信息
    json_path = OUTPUT_DIR / "pages.json"

    json_path.write_text(
        json.dumps(
            pages,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("总页数:", len(pages))

    for page in pages:
        page_num = page["page"]

        # 单页TXT便于人工检查
        txt_path = OUTPUT_DIR / f"page_{page_num:02d}.txt"
        txt_path.write_text(
            page["text"],
            encoding="utf-8",
        )

        print("\n" + "=" * 60)
        print("页码:", page_num)
        print("字符数:", len(page["text"]))
        print("保留Block:", len(page["blocks"]))
        print("过滤Block:", len(page["filtered_blocks"]))
        print("异常数量:", len(page["warnings"]))

        for warning in page["warnings"][:10]:
            print("  WARNING:", warning)

        print("文本预览:")
        print(
            page["text"][:280].replace("\n", " ")
        )

    print("\n结果目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

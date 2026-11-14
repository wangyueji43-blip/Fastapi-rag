
from pathlib import Path
import pymupdf


def extract_page_columns(page) -> str:
    """按左右两栏提取页面文本。"""

    rect = page.rect
    mid_x = rect.x0 + rect.width / 2

    left_rect = pymupdf.Rect(
        rect.x0,
        rect.y0,
        mid_x,
        rect.y1,
    )

    right_rect = pymupdf.Rect(
        mid_x,
        rect.y0,
        rect.x1,
        rect.y1,
    )

    left_text = page.get_text(
        "text",
        clip=left_rect,
        sort=True,
    )

    right_text = page.get_text(
        "text",
        clip=right_rect,
        sort=True,
    )

    return left_text + "\n\n" + right_text


def load_pdf(
    pdf_path: str,
    mode: str = "normal"
) -> list[dict]:
    """
    读取 PDF 并按页提取文本。

    mode:
        normal  - 整页提取
        columns - 左右两栏提取
    """

    if mode not in {"normal", "columns"}:
        raise ValueError(
            "mode 只能是 normal 或 columns"
        )

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 PDF 文件: {path}"
        )

    pages = []

    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc):

            if mode == "columns":
                text = extract_page_columns(page)
            else:
                text = page.get_text(
                    "text",
                    sort=True,
                )

            text = text.strip()

            if not text:
                continue

            pages.append({
                "source": path.name,
                "page": page_index + 1,
                "text": text,
            })

    return pages

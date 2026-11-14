#第一步：文本清洗 text_cleaner.py
"""先解释为什么要清洗。
你之前的 PDF 中存在：
The model was designed for cell type–
specific GRN inference.
以及：
The model uses multi-
head attention.
还有大量因为 PDF 排版产生的不规则空格。
但我们不能简单地：
text = " ".join(text.split())
因为这会删除所有换行和空行，把可能有用的段落边界一并丢掉。
我们的原则是：
清除排版噪声，但尽量保留能够反映结构的信息。
"""
import re
def clean_pdf_text(text: str) -> str:
    """
    清理PDF文本中的基础排版噪声。

    目标：
    1. 统一换行格式
    2. 处理多余空格
    3. 连接换行中的单词
    4. 保留段落空行
    """

    # 统一不同平台的换行符
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 去掉不可见的零宽字符
    text = text.replace("\u200b", "")

    # 将不间断空格换成普通空格
    text = text.replace("\u00a0", " ")

    # 清理单行中的重复空格
    text = re.sub(r"[ \t]+", " ", text)
    """
    - 空格和 \t 分别代表普通空格和制表符。
    - + 表示前面的字符出现一次或多次。
    因此它可以把连续的空格和 Tab 统一成一个空格。
    """
    # 合并换行处的连字符形式：
    # multi-\nhead -> multi-head
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])",
        "-",
        text
    )

    # 将过多空行收缩到最多一个空白行
    text = re.sub(r"\n{3,}", "\n\n", text)
    """
    表示匹配连续出现三次及以上的换行符。
    我们把它替换成两个换行符，目的是保留段落分隔，同时减少不必要的空行。
    这里暂时不做过于激进的清洗，例如直接删除数字、短文本或公式。
    """
    return text.strip()

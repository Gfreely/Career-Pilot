import os
import re
import glob
import subprocess

class PdfToMarkdownParser:
    """PDF 到 Markdown 的解析工具包，适用于简历解析等"""

    @classmethod
    def parse_pdf_with_mineru(cls, pdf_path: str, output_dir: str = None) -> str:
        """
        通过 PyMuPDF 原生读取 PDF (作为对 magic-pdf 环境崩溃的轻量级平替方案)
        避免深入嵌套的 transformers 和 layoutlmv3 模型版本冲突问题
        """
        import fitz
        
        try:
            doc = fitz.open(pdf_path)
            md_content = ""
            for i, page in enumerate(doc):
                # 提取为纯文本并做最基础的分段
                page_text = page.get_text()
                md_content += f"## Page {i + 1}\n\n{page_text}\n\n"
            
            # 由于不再调用 CLI，直接在此处返回文本
            if not md_content.strip():
                print(f"解析成功但未找到文字内容: {pdf_path}")
                return ""
            return md_content
                
        except Exception as e:
            print(f"使用 PyMuPDF 解析 PDF 时出现异常: {e}")
            return ""

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        基础文本清洗
        """
        # 去除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除 MinerU 可能产生的一些特定标记或无意义符号
        text = text.replace('•', '')
        text = text.replace('**', '')
        return text.strip()

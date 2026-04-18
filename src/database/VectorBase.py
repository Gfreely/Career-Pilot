import os
import sys
# Project root is three levels up from src/database/VectorBase.py if we want src in path, wait
# actually __file__ is src/database/VectorBase.py
# dir(__file__) is src/database
# dir(dir(__file__)) is src
# dir(dir(dir(__file__))) is project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import re
import subprocess
import shutil
import glob
from typing import List
from tqdm import tqdm

# Vector DB & Embeddings imports
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.core.embedding_model import LocalBGEM3Embeddings

class PDFVectorBase:
    def __init__(self, pdf_dir: str):
        """
        初始化 PDF 向量数据库构建器
        :param pdf_dir: PDF 文件所在目录
        """
        self.pdf_dir = pdf_dir
        self.milvus_args = {"host": "127.0.0.1", "port": "19530"}
        self.embedding = LocalBGEM3Embeddings()
        
        # Markdown 分割配置
        self.headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
            ("####", "Header_4")
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def process_and_store(self):
        """
        主逻辑：解析 PDF -> 清洗分割 -> 存储到 Chroma
        """
        from src.utils.pdf_parser import PdfToMarkdownParser
        all_split_docs = []
        
        # 筛选 PDF 文件
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"目录 {self.pdf_dir} 下未找到 PDF 文件。")
            return

        print(f"检测到 {len(pdf_files)} 个 PDF 文件，准备使用 MinerU 开始解析...")

        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            
            # 1. 命令行解析
            md_text = PdfToMarkdownParser.parse_pdf_with_mineru(pdf_path)
            if not md_text:
                continue
            
            # 2. 清洗
            cleaned_text = PdfToMarkdownParser.clean_text(md_text)
            
            # 3. 按 Markdown 标题层级初步分割
            md_header_splits = self.markdown_splitter.split_text(cleaned_text)
            
            # 4. 长度切分并添加元数据
            for doc in md_header_splits:
                # 记录来源文件
                doc.metadata["source"] = pdf_file
                # 二次切分（防止单段过长影响 Embedding 效果）
                chunks = self.text_splitter.split_documents([doc])
                all_split_docs.extend(chunks)

        if not all_split_docs:
            print("未能提取到有效文档，构建停止。")
            return

        print(f"解析完成，共生成 {len(all_split_docs)} 个文本块。正在写入 Chroma 向量库...")

        # 5. 使用统一管理器追加写入向量库
        from src.database.milvus_manager import get_milvus_client, init_or_reset_collection, insert_docs

        print("正在连接底层知识库管理器...")
        client = get_milvus_client(uri="http://127.0.0.1:19530")
        collection_name = "xinghuo_kb"

        # 获取一个样板向量以便确定维度
        print("正在拉取样板 Embedding 以确定维度...")
        sample_vec = self.embedding.embed_query(all_split_docs[0].page_content)
        dim = len(sample_vec)

        # PDF 作为后期追加入库的内容，设置 drop_old=False 保障 md 等数据安全
        init_or_reset_collection(client, collection_name, dim, drop_old=False)
        insert_docs(client, collection_name, all_split_docs, self.embedding)
        
        print(f"✅ 成功！PDF向量库已通过统一管理端点入库。")

if __name__ == "__main__":
    # 配置路径（相对于 src 目录）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_INPUT_DIR = os.path.join(BASE_DIR, "../data/kd_pdf")

    # 实例化并运行
    builder = PDFVectorBase(PDF_INPUT_DIR)
    builder.process_and_store()

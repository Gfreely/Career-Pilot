import os
import sys
import json
import hashlib
import re
from typing import List, Dict

# 动态添加项目根目录至环境变量，解决 ModuleNotFoundError 问题
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter,Language
from langchain_core.documents import Document
from src.core.embedding_model import LocalBGEM3Embeddings, get_sparse_embedding_model
from src.database.milvus_manager import get_milvus_client, init_or_reset_collection, insert_docs, delete_docs_by_domain

# 配置常量
DATA_MD_PATH = os.path.join(BASE_DIR, 'data', 'md')
STATUS_FILE = os.path.join(BASE_DIR, 'data', 'collection_status.json')
MILVUS_URI = "http://127.0.0.1:19530"
UNIFIED_COLLECTION = "kb_knowledge"

def get_file_hash(file_path: str) -> str:
    """计算单个文件的 MD5 Hash"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_folder_hash(files: List[str]) -> str:
    """计算文件夹内所有文件的整体 Hash（排序后聚合）"""
    hashes = []
    for f in sorted(files):
        hashes.append(get_file_hash(f))
    return hashlib.md5("".join(hashes).encode()).hexdigest()

def clean_text_safe(text: str) -> str:
    """保护代码块的文本清洗逻辑"""
    text = text.replace('•', '').replace('**', '')
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    cleaned_parts = []
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    for part in parts:
        if part.startswith('```'):
            cleaned_parts.append(part)
        else:
            p = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), part)
            p = p.replace('\n\n', '\n')
            cleaned_parts.append(p)
    return "".join(cleaned_parts)

def process_domain_files(client, collection_name: str, source_domain: str, 
                         files: List[str], dense_embedding_model: LocalBGEM3Embeddings,
                         sparse_embedding_model):
    """
    处理属于同一 source_domain 的所有文件。
    生成 Dense + Sparse 双向量后入库。
    """
    print(f"\n📂 正在处理知识域: {source_domain} (含 {len(files)} 个文件)")
    
    all_chunks = []
    
    # 1. 切分配置
    headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3"), ("####", "Header_4")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,chunk_size=1000, chunk_overlap=100)

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                rel_file_path = os.path.relpath(file_path, DATA_MD_PATH)
                
                # 结构化切分
                header_splits = markdown_splitter.split_text(content)
                for chunk in header_splits:
                    headers = [chunk.metadata.get(h) for h in ["Header_1", "Header_2", "Header_3", "Header_4"] if chunk.metadata.get(h)]
                    header_context = " > ".join(headers)

                    chunk.metadata.update({
                        "source": os.path.basename(file_path),
                        "full_path": rel_file_path,
                        "collection": collection_name,
                        "source_domain": source_domain
                    })
                    chunk.page_content = clean_text_safe(f"Context: {header_context}\nContent: {chunk.page_content}")
                    
                    # 递归字符进一步细分
                    doc_chunks = text_splitter.split_documents([chunk])
                    all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"  ❌ 文件处理失败 {file_path}: {e}")

    if not all_chunks:
        print(f"  ⚠ 知识域 {source_domain} 没有生成有效切片，跳过。")
        return

    # 2. 生成 Dense 向量 (SiliconFlow API)
    texts = [doc.page_content for doc in all_chunks]
    print(f"  🔢 正在生成 Dense 向量 ({len(texts)} 条, SiliconFlow API)...")
    dense_vectors = dense_embedding_model.embed_documents(texts)
    
    # 3. 生成 Sparse 向量 (本地 BGE-M3)
    print(f"  🔢 正在生成 Sparse 向量 ({len(texts)} 条, 本地 BGE-M3)...")
    sparse_vectors = sparse_embedding_model.encode_sparse(texts)
    
    # 4. 先删除该 domain 的旧数据，再追加新数据
    delete_docs_by_domain(client, collection_name, source_domain)
    insert_docs(client, collection_name, all_chunks, dense_vectors, sparse_vectors)
    print(f"  ✅ 知识域 {source_domain} 入库成功 (共 {len(all_chunks)} 个切片, Dense+Sparse 双向量)")

def main():
    print("🚀 开始知识库聚合重构 (单集合逻辑分库方案)...")
    
    if not os.path.exists(STATUS_FILE):
        status_data = {}
    else:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            status_data = json.load(f)

    # 1. 扫描并按顶级目录分组（顶级目录名即 source_domain）
    groups = {} # source_domain -> List[file_paths]
    for root, _, files in os.walk(DATA_MD_PATH):
        rel_path = os.path.relpath(root, DATA_MD_PATH)
        source_domain = rel_path.replace('\\', '/').split('/')[0] if rel_path != '.' else "root"
        
        md_files = [os.path.join(root, f) for f in files if f.endswith('.md')]
        if md_files:
            if source_domain not in groups:
                groups[source_domain] = []
            groups[source_domain].extend(md_files)

    if not groups:
        print("❌ 未发现任何文件。")
        return

    # 2. 初始化核心组件
    dense_embedding = LocalBGEM3Embeddings()
    sparse_embedding = get_sparse_embedding_model()  # 本地 BGE-M3 单例
    client = get_milvus_client(uri=MILVUS_URI)
    
    # 3. 确保统一集合存在（显式 Schema: Dense + Sparse 双向量）
    if not client.has_collection(UNIFIED_COLLECTION):
        init_or_reset_collection(client, UNIFIED_COLLECTION, drop_old=False)
        print(f"📦 已创建统一集合 {UNIFIED_COLLECTION} (显式 Dense+Sparse Schema)")
    
    updated_count = 0
    skipped_count = 0

    # 4. 逐个 source_domain 增量处理
    for source_domain, files in groups.items():
        domain_key = f"{UNIFIED_COLLECTION}::{source_domain}"
        new_hash = get_folder_hash(files)
        
        process_domain_files(client, UNIFIED_COLLECTION, source_domain, files, 
                             dense_embedding, sparse_embedding)
        status_data[domain_key] = new_hash
        updated_count += 1

    # 5. 保存状态
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, ensure_ascii=False, indent=4)

    print(f"\n✨ 重构完毕！更新知识域: {updated_count}, 跳过: {skipped_count}")

if __name__ == "__main__":
    main()

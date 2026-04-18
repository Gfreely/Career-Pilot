"""
检索与精排统一模块 (Retrieval & Reranking Module) - 多集合增强版

将 main.py 中的检索、多级策略、画像增强、Rerank 逻辑统一封装在此。
支持基于文件夹的多集合并发检索。
"""

import os
import sys
import json
import time
import math
import numpy as np
import concurrent.futures
from typing import List, Dict, Any, Optional

# 确保从 src 目录直接运行时也能正确找到 src 包
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_core.documents import Document
from FlagEmbedding import FlagReranker
from src.database.milvus_manager import get_milvus_client, initialize_hybrid_retriever
from src.core.embedding_model import LocalBGEM3Embeddings
import src.core.template as template

# ============================================================
# 全局可调参数
# ============================================================
RERANKER_MODEL_PATH = 'model/bge-reranker-v2-m3'
PROFILE_RERANK_ALPHA = 0.8    # rerank 融合权重
RERANK_TOP_N = 5              # 精排最终保留文档数
HYBRID_RECALL_K = 15          # 每个集合的混合召回数
RERANK_CANDIDATE_LIMIT = 40   # 送入精排的最大总候选文档数
MILVUS_URI = "http://127.0.0.1:19530"
UNIFIED_COLLECTION = "kb_knowledge"  # 统一知识库集合名

# ============================================================
# 模块级单例
# ============================================================
_reranker: Optional[FlagReranker] = None
_embedding_model: Optional[LocalBGEM3Embeddings] = None

def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL_PATH, use_fp16=True, device='cuda')
    return _reranker

def _get_embedding_model() -> LocalBGEM3Embeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = LocalBGEM3Embeddings()
    return _embedding_model



# ============================================================
# 1. Rerank 核心逻辑
# ============================================================

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b)) if (norm_a != 0 and norm_b != 0) else 0.0

def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores: return []
    def sigmoid(x):
        # 防止溢出的安全处理
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)
    return [sigmoid(s) for s in scores]

def rerank_documents(query: str, docs: List[Document], profile_text: str = "", 
                     profile_vec: Optional[List[float]] = None, top_n: int = RERANK_TOP_N, 
                     alpha: float = PROFILE_RERANK_ALPHA) -> List[Document]:
    if not docs: return []
    reranker = _get_reranker()
    passages = [doc.page_content for doc in docs]
    raw_scores = reranker.compute_score([[f"{query} (Context: {profile_text})", p] for p in passages], batch_size=32, max_length=1024)
    if isinstance(raw_scores, (int, float)): raw_scores = [raw_scores]
    norm_rerank = _normalize_scores(raw_scores)

    has_profile = (profile_vec is not None) or (profile_text and profile_text.strip())
    if has_profile and alpha < 1.0:
        emb_model = _get_embedding_model()
        if profile_vec is None:
            profile_vec = emb_model.embed_query(profile_text)
        doc_vectors = emb_model.embed_documents(passages)
        profile_sims = [_cosine_similarity(profile_vec, dv) for dv in doc_vectors]
    else:
        profile_sims = [0.0] * len(docs)
        alpha = 1.0

    alpha = 1.0
    for i, doc in enumerate(docs):
        doc.metadata["final_score"] = alpha * norm_rerank[i] + (1 - alpha) * profile_sims[i]

    return sorted(docs, key=lambda x: x.metadata["final_score"], reverse=True)[:top_n]

# ============================================================
# 2. 多级检索策略逻辑 (还原)
# ============================================================

def determine_retrieval_level(query: str, llm_client: Any) -> int:
    system_prompt = template.NEED_ENHANCEMENT_JUDGE_TEMPLATE.replace("{query}", query)
    result = llm_client.call_small_model(system_prompt=system_prompt).strip()
    return 2 if "2" in result else (1 if "1" in result else 0)

def extract_keywords_and_resolve(query: str, llm_client: Any) -> str:
    system_prompt = template.SINGLE_REWRITE_TEMPLATE.replace("{query}", query)
    return llm_client.call_small_model(system_prompt=system_prompt)

def rewrite_query(query: str, llm_client: Any) -> List[str]:
    system_prompt = template.MULTI_QUERY_REWRITE_TEMPLATE.replace("{query}", query)
    content = llm_client.call_small_model(system_prompt=system_prompt).strip()
    try:
        if "```json" in content: content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[-1].split("```")[0].strip()
        result = json.loads(content)
        return result if isinstance(result, list) else []
    except: return []

def generate_hyde_document(query: str, llm_client: Any) -> str:
    system_prompt = template.HYDE_TEMPLATE_XINGHUO.replace("{query}", query)
    return llm_client.call_small_model(system_prompt=system_prompt)

def build_profile_queries(profile_filter: Dict[str, Any]) -> List[str]:
    extra = []
    if profile_filter.get("tech_stack"): extra.append(" ".join(profile_filter["tech_stack"]))
    if profile_filter.get("job_preferences"): extra.append(" ".join(profile_filter["job_preferences"]))
    if profile_filter.get("major"): extra.append(profile_filter["major"])
    return extra

# ============================================================
# 3. 统一检索管线 (多集合并发版)
# ============================================================

def execute_retrieval_pipeline(
    query: str, 
    llm_client: Any, 
    profile_text: str = "", 
    profile_vec: Optional[List[float]] = None, 
    profile_filter: Optional[Dict[str, Any]] = None,
    milvus_uri: str = MILVUS_URI,
    recall_k: int = HYBRID_RECALL_K,
    rerank_top_n: int = RERANK_TOP_N,
    rerank_candidate_limit: int = RERANK_CANDIDATE_LIMIT,
    alpha: float = PROFILE_RERANK_ALPHA,
) -> Dict[str, Any]:
    
    result = {"final_docs": [], "context": "", "retrieval_level": 0, "total_candidates": 0, "status_message": ""}
    client = get_milvus_client(uri=milvus_uri)
    
    # 检查统一集合是否存在
    if not client.has_collection(UNIFIED_COLLECTION):
        result["status_message"] = f"未发现统一知识库集合 {UNIFIED_COLLECTION}"
        return result

    # 1. 意图判定与查询改写
    search_queries = [query]
    if profile_filter:
        search_queries.extend(build_profile_queries(profile_filter))
    
    try:
        level = determine_retrieval_level(query, llm_client)
        result["retrieval_level"] = level
        if level == 1:
            search_queries.append(extract_keywords_and_resolve(query, llm_client))
        elif level == 2:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_rw = executor.submit(rewrite_query, query, llm_client)
                f_hd = executor.submit(generate_hyde_document, query, llm_client)
                search_queries.extend(f_rw.result())
                search_queries.append(f_hd.result())
    except Exception as e:
        print(f"策略准备异常: {e}")

    # 2. 单集合统一混合召回
    all_candidates = []
    seen_content = set()
    
    try:
        retriever = initialize_hybrid_retriever(client, UNIFIED_COLLECTION, k=recall_k)
        if retriever:
            for q in search_queries:
                if q:
                    docs = retriever.invoke(q)
                    for doc in docs:
                        if doc.page_content not in seen_content:
                            seen_content.add(doc.page_content)
                            all_candidates.append(doc)
    except Exception as e:
        print(f"统一集合检索异常: {e}")

    result["total_candidates"] = len(all_candidates)
    
    # 3. 筛选与精排
    candidate_pool = all_candidates[:rerank_candidate_limit]
    final_docs = rerank_documents(query, candidate_pool, profile_text, profile_vec, rerank_top_n, alpha)
    
    result["final_docs"] = final_docs
    result["context"] = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(final_docs)])
    result["status_message"] = f"统一知识库检索完成 (候选 {len(all_candidates)} 篇)"
    
    return result

if __name__ == "__main__":
    from src.core.llm_client import UnifiedLLMClient
    llm = UnifiedLLMClient()
    res = execute_retrieval_pipeline("Redis 的持久化机制有哪些？", llm)
    print(f"检索到 {len(res['final_docs'])} 条文档内容。")
    for d in res['final_docs']:
        print(d.page_content[:50])

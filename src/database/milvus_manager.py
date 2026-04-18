import time
from typing import List, Any, Optional, Dict
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from src.core.embedding_model import LocalBGEM3Embeddings, get_sparse_embedding_model

# 混合召回器缓存（避免每次请求都重新实例化）
_retriever_cache: dict = {}

# BGE-M3 Dense 向量维度
DENSE_DIM = 1024

# ----------------------------------------------------
# 1. 底层连接与表管理逻辑
# ----------------------------------------------------

def get_milvus_client(uri: str = "http://127.0.0.1:19530") -> MilvusClient:
    """获取原生的 MilvusClient，并包含连接失败的基本错误回显"""
    try:
        client = MilvusClient(uri=uri)
        return client
    except Exception as e:
        print(f"❌ Milvus 客户端初始化失败: {e}")
        raise e


def init_or_reset_collection(client: MilvusClient, collection_name: str, 
                              dim: int = DENSE_DIM, drop_old: bool = False):
    """
    初始化表结构（显式 Schema，支持 Dense + Sparse 双向量字段）。
    如果 drop_old=True，则遇到旧表时会彻底删除并重建。
    如果 drop_old=False，遇到旧表时则保留（作为 Append）。
    """
    if client.has_collection(collection_name):
        if drop_old:
            print(f"检测到旧集合 {collection_name} 且要求覆写，正在删除重建...")
            client.drop_collection(collection_name)
        else:
            print(f"数据库中已存在集合 {collection_name}，追加模式，保留现有数据...")
            return

    # ===== 构建显式 Schema =====
    print(f"正在建立集合 {collection_name}，Dense 维度: {dim} ...")
    
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    
    # 主键
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    # 文本内容
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    # Dense 向量 (BGE-M3 SiliconFlow API, 1024 维)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    # Sparse 向量 (BGE-M3 本地推理, 无需指定维度)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    
    # ===== 构建索引 =====
    index_params = client.prepare_index_params()
    
    # Dense 向量索引
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    # Sparse 向量索引
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    
    # 创建集合并加载
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"✅ 集合 {collection_name} 创建成功 (Dense + Sparse 双向量)")


def insert_docs(client: MilvusClient, collection_name: str, docs: List[Document],
                dense_vectors: List[List[float]], sparse_vectors: List[dict],
                batch_size: int = 500):
    """
    分批将文档及其 Dense / Sparse 双向量写入 Milvus。
    """
    if not docs:
        print("❌ 传入文档为空，跳过入库！")
        return
    if len(docs) != len(dense_vectors) or len(docs) != len(sparse_vectors):
        print(f"❌ 文档数({len(docs)})与向量数(dense={len(dense_vectors)}, sparse={len(sparse_vectors)})不匹配！")
        return

    start_time = time.time()

    data_to_insert = []
    for i, doc in enumerate(docs):
        entry = {
            "text": doc.page_content,
            "dense_vector": dense_vectors[i],
            "sparse_vector": sparse_vectors[i],
        }
        if doc.metadata:
            entry.update(doc.metadata)
        data_to_insert.append(entry)

    print(f"准备入库 {len(data_to_insert)} 条数据至集合 '{collection_name}'...")
    for i in range(0, len(data_to_insert), batch_size):
        batch_data = data_to_insert[i: i + batch_size]
        client.insert(
            collection_name=collection_name,
            data=batch_data
        )
        print(f"  已插入 {min(i + batch_size, len(data_to_insert))}/{len(data_to_insert)} 条...")

    print(f"✅ 入库成功！耗时 {time.time() - start_time:.2f}s")
    
    # 入库后清除召回器缓存，确保新数据生效
    invalidate_retriever_cache(collection_name)


def delete_docs_by_domain(client: MilvusClient, collection_name: str, source_domain: str):
    """
    按 source_domain 删除指定知识域的全部文档，用于增量更新前的清理。
    """
    try:
        result = client.delete(
            collection_name=collection_name,
            filter=f'source_domain == "{source_domain}"'
        )
        print(f"🗑️ 已从 {collection_name} 中删除 source_domain='{source_domain}' 的文档")
        return result
    except Exception as e:
        print(f"❌ 删除 source_domain='{source_domain}' 文档失败: {e}")
        return None


def invalidate_retriever_cache(collection_name: str = None):
    """
    清除混合召回器缓存。
    - 指定 collection_name 时只清除对应缓存
    - 不指定时清除所有缓存
    """
    global _retriever_cache
    if collection_name:
        keys_to_remove = [k for k in _retriever_cache if k.startswith(collection_name)]
        for k in keys_to_remove:
            del _retriever_cache[k]
        if keys_to_remove:
            print(f"[Cache] 已清除集合 '{collection_name}' 的召回器缓存")
    else:
        _retriever_cache.clear()
        print("[Cache] 已清除所有召回器缓存")


# ----------------------------------------------------
# 2. Milvus 原生混合检索封装
# ----------------------------------------------------

class NativeHybridRetriever(BaseRetriever):
    """
    基于 Milvus 原生 hybrid_search 的混合召回器。
    同时执行 Dense ANN + Sparse ANN，使用 WeightedRanker 融合结果。
    
    替代旧方案中 "拉取全量文档 → 本地 BM25 → EnsembleRetriever" 的笨重链路。
    """
    client: Any
    collection_name: str
    dense_embedding_model: Any
    sparse_embedding_model: Any
    top_k: int = 50
    sparse_weight: float = 0.7
    dense_weight: float = 0.3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. 并行计算 dense 和 sparse 查询向量
        dense_query_vec = self.dense_embedding_model.embed_query(query)
        sparse_query_vec = self.sparse_embedding_model.encode_query_sparse(query)

        # 2. 构建双路 ANN 搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_query_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=self.top_k,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_query_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=self.top_k,
        )

        # 3. Milvus 原生混合搜索 + 加权融合
        try:
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[sparse_req, dense_req],
                ranker=WeightedRanker(self.sparse_weight, self.dense_weight),
                limit=self.top_k,
                output_fields=["text", "source", "source_domain"],
            )

            docs = []
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {})
                    text = entity.pop("text", "")
                    docs.append(Document(page_content=text, metadata=entity))
            return docs

        except Exception as e:
            print(f"❌ Milvus hybrid_search 失败: {e}")
            return []


def initialize_hybrid_retriever(client: MilvusClient, collection_name: str, 
                                 k: int = 50, force_rebuild: bool = False) -> Any:
    """
    初始化并对外暴露 Milvus 原生混合召回器。
    使用模块级缓存避免重复实例化 embedding 模型。
    
    接口签名与旧版保持一致，对上层（Recall_test.py 等）透明。
    """
    cache_key = f"{collection_name}_{k}"
    
    if not force_rebuild and cache_key in _retriever_cache:
        print(f"[Cache] 使用缓存的混合召回器 (key={cache_key})")
        return _retriever_cache[cache_key]
    
    if client is None:
        return None
    
    if not client.has_collection(collection_name):
        print(f"警告：集合 {collection_name} 不存在，无法构建召回器。")
        return None

    # 获取 embedding 模型实例
    dense_model = LocalBGEM3Embeddings()
    sparse_model = get_sparse_embedding_model()
    
    retriever = NativeHybridRetriever(
        client=client,
        collection_name=collection_name,
        dense_embedding_model=dense_model,
        sparse_embedding_model=sparse_model,
        top_k=k,
        sparse_weight=0.7,
        dense_weight=0.3,
    )
    
    _retriever_cache[cache_key] = retriever
    print(f"[Cache] Milvus 原生混合召回器已构建并缓存 (key={cache_key})")
    return retriever

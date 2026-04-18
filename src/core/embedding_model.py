from typing import List
from langchain_core.embeddings import Embeddings
import os

# ============================================================
# 本地 BGE-M3 模型加载器（单例）
# ============================================================

_bgem3_model_instance = None

def get_bgem3_model(device: str = "cuda"):
    """
    获取 BGE-M3 模型单例。
    模型默认路径为项目根目录下的 model/bge-m3。
    """
    global _bgem3_model_instance
    if _bgem3_model_instance is None:
        from FlagEmbedding import BGEM3FlagModel
        
        # 动态计算绝对路径：当前文件在 src/core/，根目录为上两级目录
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "model", "bge-m3")
        
        print(f"🔄 正在加载本地 BGE-M3 模型 ({model_path})...")
        # 采用 fp16 加载可以有效控制显存在2.3GB左右
        _bgem3_model_instance = BGEM3FlagModel(model_path, use_fp16=True, device=device)
        print(f"✅ BGE-M3 模型加载完毕 (device={device})")
        
    return _bgem3_model_instance

class LocalBGEM3Embeddings(Embeddings):
    """
    本地 BGE-M3 稠密向量嵌入，兼容 LangChain Embeddings 接口。
    """
    def __init__(self, device: str = "cuda"):
        # 预先触发加载
        self.device = device
        get_bgem3_model(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = get_bgem3_model(self.device)
        # BGE-M3FlagModel encode返回字典
        output = model.encode(
            texts, 
            batch_size=12,
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        import numpy as np
        if isinstance(output['dense_vecs'], np.ndarray):
            return output['dense_vecs'].tolist()
        return output['dense_vecs']

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ============================================================
# 本地 BGE-M3 Sparse Embedding（单例）
# ============================================================

_sparse_instance = None

class LocalBGEM3SparseEmbeddings:
    """
    使用本地 BGE-M3 模型生成稀疏向量 (learned sparse embedding)。
    稀疏向量的每个维度对应一个 Token ID，值为该 Token 的重要性权重。
    输出格式 Dict[int, float] 直接兼容 Milvus SPARSE_FLOAT_VECTOR。
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        get_bgem3_model(self.device)
    
    def encode_sparse(self, texts: List[str], batch_size: int = 12) -> list:
        model = get_bgem3_model(self.device)
        output = model.encode(
            texts, 
            batch_size=batch_size,
            return_dense=False, 
            return_sparse=True, 
            return_colbert_vecs=False,
        )
        sparse_vecs = []
        for sparse_dict in output["lexical_weights"]:
            converted = {int(k): float(v) for k, v in sparse_dict.items()}
            sparse_vecs.append(converted)
        return sparse_vecs
    
    def encode_query_sparse(self, text: str) -> dict:
        return self.encode_sparse([text])[0]

def get_sparse_embedding_model(device: str = "cuda") -> LocalBGEM3SparseEmbeddings:
    """获取 BGE-M3 稀疏嵌入模型的全局单例"""
    global _sparse_instance
    if _sparse_instance is None:
        _sparse_instance = LocalBGEM3SparseEmbeddings(device=device)
    return _sparse_instance


import sys
import threading
import time
import os
from pymilvus import connections, utility, MilvusClient, MilvusException

# 将当前脚本的父目录（即项目根目录）加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embedding_model import LocalBGEM3Embeddings
from langchain_core.documents import Document

# 设置硬超时时间 30 秒
TIMEOUT = 30

def timeout_handler():
    print("\n" + "="*60)
    print("❌ 测试超时 (30秒)！脚本被强制中断！")
    print("可能原因：")
    print("1. 宿主机与 Docker 容器间的 gRPC 握手死锁（通常发生在 Windows 重启后 Docker 未就绪）。")
    print("2. SiliconFlow API 响应极慢导致 embed_documents 阻塞。")
    print("="*60)
    os._exit(1)

def test_milvus_health():
    print("="*60)
    print("🚀 开始 Milvus Docker 稳定性与连接句柄检测...")
    
    timer = threading.Timer(TIMEOUT, timeout_handler)
    timer.start()

    try:
        # ---- 1. 底层 MilvusClient 连通性探测 (最稳健的方法) ----
        print("[1/4] 正在探测 MilvusClient 句柄 (127.0.0.1:19530) ...")
        try:
            # 现代版本推荐使用 MilvusClient 直接初始化
            client = MilvusClient(uri="http://127.0.0.1:19530")
            # 尝试执行一个极轻量的操作：列出集合
            collections_list = client.list_collections()
            print(f"✅ MilvusClient 握手成功！当前集合数: {len(collections_list)}")
        except Exception as e:
            print(f"❌ MilvusClient 无法建立稳定连接: {e}")
            return

        # ---- 2. Embedding 引擎加载 ----
        print("[2/4] 正在加载本地 BGE-M3 词向量引擎...")
        embedding = LocalBGEM3Embeddings()
        try:
            test_vec = embedding.embed_query("connection test")
            print(f"✅ Embedding 引擎正常 (维度: {len(test_vec)})")
        except Exception as e:
            print(f"❌ Embedding 引擎失效，请检查 API Key 或网络: {e}")
            return

        # ---- 3. 模拟 LangChain 写入流程 (带断联防御) ----
        collection_name = "test_stability_kb"
        print("[3/4] 正在执行数据灌入与持久化测试...")
        
        try:
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)

            # 准备测试数据
            test_content = "Docker Milvus 稳定性测试数据"
            vector = embedding.embed_query(test_content)
            
            # 使用 MilvusClient 执行插入，这是绕过幽灵断联的终极手段
            client.create_collection(
                collection_name=collection_name,
                dimension=len(test_vec),
                metric_type="L2",
                auto_id=True
            )
            
            client.insert(
                collection_name=collection_name,
                data=[{"vector": vector, "text": test_content}]
            )
            print(f"✅ 数据底层写入成功！")

        except Exception as e:
            print(f"❌ 写入阶段发生异常: {e}")
            return
            
        # ---- 4. 检索验证 ----
        try:
            print("[4/4] 正在执行相似度检索验证...")
            search_res = client.search(
                collection_name=collection_name,
                data=[vector],
                limit=1,
                output_fields=["text"]
            )
            if len(search_res) > 0:
                print(f"✅ 检索回显正常: {search_res[0][0]['entity']['text']}")
                print("\n🎉 结论：您的 Milvus 容器和 Embedding 接口完全正常！")
                print("💡 建议：请在生产代码中改用 MilvusClient 或更新 langchain_milvus 库。")
            
        except Exception as e:
            print(f"❌ 检索验证失败: {e}")
            
    finally:
        timer.cancel()
        print("="*60)

if __name__ == "__main__":
    test_milvus_health()
from pymilvus import connections, utility, Collection

# 连接到 Milvus
def connect_to_milvus(host="127.0.0.1", port="19530"):
    connections.connect("default", host=host, port=port)
    print(f"已连接到 Milvus 服务: {host}:{port}")

# 获取所有 collections 的名称
def list_collections():
    collection_names = utility.list_collections()
    print("当前 Milvus 中的 collections:", collection_names)
    return collection_names

# 删除指定 collection
def drop_collection(collection_name):
    from pymilvus import Collection
    collection = Collection(collection_name)
    collection.drop()
    print(f"集合 '{collection_name}' 已成功删除")

# 删除所有 collections
def drop_all_collections():
    collection_names = list_collections()
    for collection_name in collection_names:
        drop_collection(collection_name)

# 主函数
def main():
    # 连接到 Milvus
    connect_to_milvus()

    # 删除所有 collections
    drop_all_collections()

if __name__ == "__main__":
    main()

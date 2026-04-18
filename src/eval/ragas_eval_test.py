import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 导入检索管线
from src.eval.Recall_test import execute_retrieval_pipeline # 确保模块名与你之前修改的一致
from src.core.llm_client import UnifiedLLMClient

# Ragas 相关导入
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, context_entity_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 配置评估参数
EVAL_DATA_PATH = os.path.join(BASE_DIR, "data", "ragas_eval_data.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "ragas_eval_results.csv")

def load_eval_data(path: str, limit: int = None, shuffle: bool = True) -> List[Dict[str, Any]]:
    """加载评估数据集"""
    data = []
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # 1. 执行打乱操作
    if shuffle:
        print(f"🎲 正在打乱数据集（总计 {len(data)} 条）...")
        random.seed(42)  # 设置随机种子，保证实验可重复性
        random.shuffle(data)
    return data[:limit]

def run_evaluation(limit: int):
    print(f"🚀 开始 RAG 检索链路评估 (样本数: {limit})...")
    
    # 1. 加载数据
    raw_eval_items = load_eval_data(EVAL_DATA_PATH, limit)
    if not raw_eval_items:
        print(f"❌ 未找到评估数据: {EVAL_DATA_PATH}，请先生成数据集。")
        return

    # 2. 初始化 LLM 客户端
    llm_client = UnifiedLLMClient()
    
    # 3. 执行检索管线推理
    # Ragas 0.2+ 要求的标准列表
    questions = []
    contexts = []
    ground_truths = []
    
    print("🔍 正在通过检索管线获取实时结果...")
    for item in tqdm(raw_eval_items):
        question = item["question"]
        # 注意：ground_truth 必须是字符串，代表标准答案
        gt_answer = item.get("ground_truth", "") 
        
        try:
            # 调用你之前的统一检索模块
            res = execute_retrieval_pipeline(
                query=question, 
                llm_client=llm_client,
                # 如果有画像信息，可以从 item 中提取传入
                profile_text=item.get("metadata", {}).get("profile", "") 
            )
            
            # 提取检索到的内容列表 (List[str])
            retrieved_docs = [doc.page_content for doc in res.get("final_docs", [])]
                
            questions.append(question)
            contexts.append(retrieved_docs)
            # 修正点：ground_truths 列表的元素必须是字符串
            ground_truths.append(str(gt_answer)) 

        except Exception as e:
            print(f"\n⚠️ 处理问题 '{question[:20]}...' 时出错: {e}")
            continue

    # 4. 构建 Ragas 数据集
    dataset_dict = {
        "question": questions,
        "contexts": contexts,
        "ground_truth": ground_truths # Ragas 内部会自动对应到 reference
    }
    dataset = Dataset.from_dict(dataset_dict)

    # 5. 配置评估模型 (SiliconFlow)
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 SILICONFLOW_API_KEY 环境变量。")
        return

    # 建议使用 72B 或更高等级模型进行评估，保证逻辑判断准确
    eval_llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct", 
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
    )
    
    eval_embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        # 兼容性设置
        check_embedding_ctx_length=False 
    )

    # 6. 执行评估
    print(f"\n📊 正在计算指标 (样本量: {len(questions)})...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision, 
                context_recall,
                context_entity_recall,
                ],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )

        # 7. 整理结果并保存
        df = result.to_pandas()
        df.to_csv(RESULTS_PATH, index=False, encoding='utf-8-sig')
        
        print("\n✅ 评估完成！")
        print("-" * 30)
        print(f"指标平均得分:\n{result}")
        print("-" * 30)
        print(f"详细结果已保存至: {RESULTS_PATH}")
        
    except Exception as e:
        print(f"❌ 评估计算阶段崩溃: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="评估样本数量限制")
    args = parser.parse_args()
    
    run_evaluation(args.limit)
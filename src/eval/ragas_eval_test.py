import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import random

# 脚本位于 src/eval/，需要向上三级才能得到项目根目录（xinghuollm/）
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 导入检索管线
from src.eval.Recall_test import execute_retrieval_pipeline
from src.core.llm_client import UnifiedLLMClient
from src.core.template import RAG_TEMPLATE_XINGHUO

# Ragas 相关导入
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ============ 配置评估参数 ============
EVAL_DATA_PATH = os.path.join(BASE_DIR, "data", "ragas_eval_data.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "ragas_eval_results.csv")

# 指标组合定义
METRIC_GROUPS = {
    "retrieval":   [context_precision, context_recall],
    "generation":  [faithfulness],
    "full":        [context_precision, context_recall, faithfulness],
}


def load_eval_data(path: str, limit: int = None, shuffle: bool = True) -> List[Dict[str, Any]]:
    """加载评估数据集（支持含/不含 answer 字段的两种格式）"""
    data = []
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if shuffle:
        print(f"🎲 正在打乱数据集（总计 {len(data)} 条）...")
        random.seed(42)
        random.shuffle(data)
    return data[:limit]


def generate_answer(question: str, contexts: List[str], llm_client: UnifiedLLMClient, model_name: str) -> str:
    """
    基于已召回的 contexts，调用大模型生成 RAG 回答。
    当数据集中没有预生成的 answer 字段时使用此函数。
    使用 stream=False 非流式调用，确保可以完整收集结果。
    """
    if not contexts:
        return ""

    # 拼接上下文（与 Recall_test.py 中格式保持一致）
    context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    prompt = RAG_TEMPLATE_XINGHUO.format(context=context_text, query=question)

    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_client.call_large_model(
            messages=messages,
            model_name=model_name,
            stream=False,
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"\n⚠️ 生成 answer 失败 (question: '{question[:30]}...'): {e}")
        return ""


def run_evaluation(limit: int, mode: str = "full", gen_model: str = "Qwen/Qwen2.5-72B-Instruct"):
    """
    主评估函数。

    Args:
        limit:     评估样本数量上限
        mode:      评测模式 (retrieval | generation | full)
        gen_model: 当数据集缺少 answer 字段时，用于在线生成 answer 的模型名称
    """
    print(f"🚀 开始 RAG 链路评估 (样本数: {limit}, 模式: {mode})...")
    print(f"   📐 将计算指标: {[m.name for m in METRIC_GROUPS[mode]]}")

    # 1. 加载数据
    raw_eval_items = load_eval_data(EVAL_DATA_PATH, limit)
    if not raw_eval_items:
        print(f"❌ 未找到评估数据: {EVAL_DATA_PATH}，请先生成数据集。")
        return

    # 2. 初始化 LLM 客户端
    llm_client = UnifiedLLMClient()

    # 3. 判断数据集是否已含 answer 字段
    has_prebuilt_answers = all("answer" in item and item["answer"] for item in raw_eval_items[:5])
    needs_answer = mode in ("generation", "full")

    if needs_answer and not has_prebuilt_answers:
        print(f"⚠️ 数据集中缺少 answer 字段，将通过检索管线 + LLM（{gen_model}）实时生成...")
        print(f"   （建议先运行 generate_eval_dataset.py 生成含 answer 的数据集以提升效率）")
        needs_retrieval = True
    elif needs_answer and has_prebuilt_answers:
        print(f"✅ 检测到预生成的 answer 字段，直接使用，跳过在线生成。")
        needs_retrieval = False
    else:
        # retrieval-only 模式，不需要 answer 也不需要 retrieval（直接用 contexts 字段）
        needs_retrieval = False

    # 4. 构建评估列表
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []

    print(f"\n🔍 正在准备评测数据...")

    for item in tqdm(raw_eval_items):
        question = item["question"]
        gt_answer = item.get("ground_truth", "")
        # 数据集中已有的 contexts 片段（用于 precision/recall 计算）
        prebuilt_contexts = item.get("contexts", [])

        try:
            if needs_retrieval:
                # 通过检索管线获取实时召回结果 + 生成 answer
                res = execute_retrieval_pipeline(
                    query=question,
                    llm_client=llm_client,
                    profile_text=item.get("metadata", {}).get("profile", "")
                )
                retrieved_docs = [doc.page_content for doc in res.get("final_docs", [])]
                # 使用实时召回结果的 contexts 来生成 answer
                answer = generate_answer(question, retrieved_docs, llm_client, gen_model)
                ctx = retrieved_docs
            elif needs_answer and has_prebuilt_answers:
                # 使用数据集中预生成的 answer 和 contexts
                answer = item.get("answer", "")
                ctx = prebuilt_contexts
            else:
                # retrieval 模式：直接使用数据集中的 contexts，answer 留空
                answer = ""
                ctx = prebuilt_contexts

            if not ctx:
                # 没有任何召回内容，跳过
                continue

            questions.append(question)
            contexts_list.append(ctx)
            answers.append(answer)
            ground_truths.append(str(gt_answer))

        except Exception as e:
            print(f"\n⚠️ 处理问题 '{question[:30]}...' 时出错: {e}")
            continue

    # 5. 构建 Ragas Dataset
    dataset_dict: Dict[str, Any] = {
        "question": questions,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    if needs_answer:
        dataset_dict["answer"] = answers

    dataset = Dataset.from_dict(dataset_dict)

    # 6. 配置评估模型（SiliconFlow）
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置 SILICONFLOW_API_KEY 环境变量。")
        return

    # 使用 72B 以上模型作为 Judge，保证逻辑判断准确
    eval_llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
    )

    eval_embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )

    # 7. 执行评估
    selected_metrics = METRIC_GROUPS[mode]
    print(f"\n📊 正在计算指标（样本量: {len(questions)}）...")
    print(f"   指标列表: {[m.name for m in selected_metrics]}")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
        )

        # 8. 整理结果并保存
        df = result.to_pandas()

        # 结果文件名根据 mode 区分，避免覆盖
        results_path = RESULTS_PATH.replace(".csv", f"_{mode}.csv")
        df.to_csv(results_path, index=False, encoding="utf-8-sig")

        print("\n✅ 评估完成！")
        print("-" * 50)
        print(f"📈 指标平均得分（mode={mode}）:")
        for metric in selected_metrics:
            raw = result[metric.name]
            # 新版 ragas 返回 list（每条样本的分数），旧版返回 float
            if isinstance(raw, (list, tuple)):
                scores_arr = np.array([s for s in raw if s is not None], dtype=float)
                valid_n    = int(np.sum(~np.isnan(scores_arr)))
                avg_score  = float(np.nanmean(scores_arr)) if valid_n > 0 else float("nan")
                note       = f"  (有效样本 {valid_n}/{len(raw)})" if valid_n < len(raw) else ""
            else:
                avg_score = float(raw)
                note      = ""

            if np.isnan(avg_score):
                print(f"   {metric.name:<30}: N/A  ⚠️ 所有样本评分失败{note}")
            else:
                print(f"   {metric.name:<30}: {avg_score:.4f}{note}")
        print("-" * 50)
        print(f"💾 详细结果已保存至: {results_path}")

    except Exception as e:
        print(f"❌ 评估计算阶段崩溃: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Ragas 评测脚本")
    parser.add_argument(
        "--limit", type=int, default=50,
        help="评估样本数量限制（默认: 50）"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["retrieval", "generation", "full"],
        help=(
            "评测模式：\n"
            "  retrieval  - 仅检索端指标（context_precision + context_recall）\n"
            "  generation - 仅生成端指标（faithfulness）\n"
            "  full       - 全链路评测（默认，三项指标全评）"
        )
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-72B-Instruct",
        help="当数据集缺少 answer 字段时，在线生成 answer 所用的大模型（默认: Qwen/Qwen2.5-72B-Instruct）"
    )
    args = parser.parse_args()

    run_evaluation(limit=args.limit, mode=args.mode, gen_model=args.model)
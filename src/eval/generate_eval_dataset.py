import os
import sys
import json
import re
import time
import concurrent.futures
from typing import List, Dict, Optional
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.llm_client import UnifiedLLMClient
from src.core.template import (
    EVAL_CHUNK_QA_TEMPLATE,
    EVAL_CROSS_CHAPTER_QA_TEMPLATE,
    EVAL_SUMMARY_TEMPLATE,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ============ 配置项 ============
DATA_MD_PATH = BASE_DIR / 'data' / 'md'
OUTPUT_FILE = BASE_DIR / 'data' / 'ragas_eval_data.jsonl'

EVAL_MODEL = "Qwen/Qwen3.5-35B-A3B"     # 评测数据集生成专用模型
MAX_CHUNKS_PER_FILE = 5                   # 单文件最大采样分块数
MAX_WORKERS = 3                           # 并发线程数


# ============ 模型调用 ============

def call_eval_model(llm_client: UnifiedLLMClient, prompt: str) -> str:
    """
    调用评测专用大模型，关闭思考模式 (Qwen3.5 no_think)。
    SiliconFlow API 要求 messages 中必须包含 user 角色消息，
    因此将 prompt 放入 user 消息，system 仅保留角色定义。
    """
    messages = [
        {"role": "system", "content": "你是一个专业的 RAG 评测数据集生成专家。请严格按照用户要求的格式输出。"},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_client.client.chat.completions.create(
            messages=messages,
            model=EVAL_MODEL,
            stream=False,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n⚠️ 评测模型调用异常 [{EVAL_MODEL}]: {e}")
        return ""


# ============ 文档分块提取 ============

def extract_chunks(file_path: Path, max_chunks: int = MAX_CHUNKS_PER_FILE) -> List[str]:
    """
    提取文件的多个有效内容块。
    通过 MarkdownHeaderTextSplitter 按标题拆分，筛选长度合适的分块，
    并在超过上限时进行均匀间隔采样，保证覆盖面。
    """
    try:
        content = file_path.read_text(encoding='utf-8')

        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        splits = splitter.split_text(content)

        if not splits:
            # 无法按标题拆分时，取整篇内容（截断）
            return [content[:1500]] if len(content) > 50 else []

        # 筛选长度合适的分块（200~1500 字符）
        valid_chunks = []
        for chunk in splits:
            text = chunk.page_content.strip()
            if 200 < len(text) < 1500:
                valid_chunks.append(text)

        # 如果没有合格分块，放宽条件取最长的分块
        if not valid_chunks:
            longest = max(splits, key=lambda c: len(c.page_content))
            text = longest.page_content.strip()
            if len(text) > 50:
                return [text[:1500]]
            return []

        # 均匀采样：超过上限时间隔取样，保证分布均匀
        if len(valid_chunks) > max_chunks:
            step = len(valid_chunks) / max_chunks
            valid_chunks = [valid_chunks[int(i * step)] for i in range(max_chunks)]

        return valid_chunks
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return []


# ============ 文档摘要生成 ============

def generate_document_summary(llm_client: UnifiedLLMClient, file_path: Path) -> str:
    """生成文档的整体摘要，用于跨章节出题。截取前 4000 字符以避免溢出上下文。"""
    try:
        content = file_path.read_text(encoding='utf-8')
        truncated = content[:4000]

        prompt = EVAL_SUMMARY_TEMPLATE.format(content=truncated)
        summary = call_eval_model(llm_client, prompt)
        return summary
    except Exception as e:
        print(f"生成摘要失败 {file_path}: {e}")
        return ""


# ============ JSON 解析 ============

def parse_qa_json(response: str) -> List[Dict[str, str]]:
    """
    从模型响应中解析 JSON 格式的 QA 对。
    先尝试直接解析，失败后再尝试提取被 Markdown 代码块包裹的 JSON。
    """
    if not response:
        return []

    def _validate(qa_list):
        """校验并提取有效的 QA 对"""
        if not isinstance(qa_list, list):
            return []
        return [
            {"question": item["question"].strip(), "answer": item["answer"].strip()}
            for item in qa_list
            if isinstance(item, dict)
            and "question" in item and "answer" in item
            and len(item["question"].strip()) > 5
            and len(item["answer"].strip()) > 5
        ]

    # 策略一：直接整体解析
    try:
        result = _validate(json.loads(response))
        if result:
            return result
    except json.JSONDecodeError:
        pass

    # 策略二：提取第一个 [...] JSON 数组块
    json_match = re.search(r'\[.*\]', response, re.S)
    if json_match:
        try:
            result = _validate(json.loads(json_match.group()))
            if result:
                return result
        except json.JSONDecodeError:
            pass

    print(f"  ⚠️ JSON 解析失败，跳过该响应")
    return []


# ============ QA 生成 ============

def generate_chunk_qa(llm_client: UnifiedLLMClient, chunk_content: str, file_name: str) -> List[Dict[str, str]]:
    """基于单个 chunk 生成简单/中等难度的 QA 对"""
    if not chunk_content or len(chunk_content) < 50:
        return []

    prompt = EVAL_CHUNK_QA_TEMPLATE.format(
        file_name=file_name,
        chunk_content=chunk_content,
    )
    response = call_eval_model(llm_client, prompt)
    return parse_qa_json(response)


def generate_cross_chapter_qa(llm_client: UnifiedLLMClient, summary: str, file_name: str) -> List[Dict[str, str]]:
    """基于文档摘要生成跨章节的较难 QA 对"""
    if not summary or len(summary) < 100:
        return []

    prompt = EVAL_CROSS_CHAPTER_QA_TEMPLATE.format(
        file_name=file_name,
        summary=summary,
    )
    response = call_eval_model(llm_client, prompt)
    return parse_qa_json(response)


# ============ 单文件处理流水线 ============

def process_file(file_path: Path, llm_client: UnifiedLLMClient) -> List[Dict]:
    """
    处理单个文件，返回多层次的评测数据项。
    第一层：基于单个 chunk 的简单题（difficulty=simple）
    第二层：基于文档摘要的跨章节题（difficulty=cross_chapter）
    """
    rel_path = file_path.relative_to(DATA_MD_PATH)
    file_name = file_path.name
    source_domain = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"

    final_items = []

    # ===== 第一层：基于单个 chunk 的简单题 =====
    chunks = extract_chunks(file_path)
    for i, chunk in enumerate(chunks):
        qa_results = generate_chunk_qa(llm_client, chunk, file_name)
        for qa in qa_results:
            item = {
                "question": qa["question"],
                "contexts": [chunk],
                "ground_truth": qa["answer"],
                "metadata": {
                    "source_file": file_name,
                    "full_path": str(rel_path),
                    "source_domain": source_domain,
                    "difficulty": "simple",
                    "chunk_index": i,
                },
            }
            final_items.append(item)

    # ===== 第二层：基于文档摘要的跨章节难题 =====
    summary = generate_document_summary(llm_client, file_path)
    if summary:
        cross_qa = generate_cross_chapter_qa(llm_client, summary, file_name)
        for qa in cross_qa:
            item = {
                "question": qa["question"],
                "contexts": [summary],
                "ground_truth": qa["answer"],
                "metadata": {
                    "source_file": file_name,
                    "full_path": str(rel_path),
                    "source_domain": source_domain,
                    "difficulty": "cross_chapter",
                },
            }
            final_items.append(item)

    return final_items


# ============ 主入口 ============

def main():
    print("🚀 启动评测数据集生成流程...")
    print(f"📌 使用模型: {EVAL_MODEL} (思考模式已关闭)")
    print(f"📌 单文件最大采样分块数: {MAX_CHUNKS_PER_FILE}")
    llm_client = UnifiedLLMClient()

    # 搜集文件
    all_files = list(DATA_MD_PATH.rglob('*.md'))
    total_files = len(all_files)
    print(f"📂 发现 {total_files} 个 Markdown 文件。\n")

    processed_count = 0
    total_qa_count = 0
    simple_count = 0
    cross_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(process_file, f, llm_client): f for f in all_files}

            for future in concurrent.futures.as_completed(future_to_file):
                processed_count += 1
                source_file = future_to_file[future]
                try:
                    items = future.result()
                    if items:
                        for item in items:
                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            total_qa_count += 1
                            if item["metadata"]["difficulty"] == "simple":
                                simple_count += 1
                            else:
                                cross_count += 1
                        f_out.flush()

                    print(
                        f"进度: [{processed_count}/{total_files}] "
                        f"| 总 QA: {total_qa_count} (简单: {simple_count}, 跨章节: {cross_count}) "
                        f"| 当前: {source_file.name}",
                        end='\r',
                    )
                except Exception as e:
                    print(f"\n❌ 处理出错 [{source_file.name}]: {e}")

    print(f"\n\n✨ 任务完成！")
    print(f"📊 统计: 处理文件 {processed_count} 个")
    print(f"   - 简单题 (chunk-level):    {simple_count} 条")
    print(f"   - 跨章节题 (cross-chapter): {cross_count} 条")
    print(f"   - 总计:                     {total_qa_count} 条")
    print(f"💾 结果保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
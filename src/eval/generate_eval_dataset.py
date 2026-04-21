import os
import sys
import json
import re
import time
import threading
import concurrent.futures
from typing import List, Dict, Optional, Set
from pathlib import Path

# 基础路径配置：脚本位于 src/eval/，需要向上三级才能得到项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # xinghuollm/
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.llm_client import UnifiedLLMClient
from src.core.template import (
    EVAL_CHUNK_QA_TEMPLATE,
    EVAL_CROSS_CHAPTER_QA_TEMPLATE,
    EVAL_SUMMARY_TEMPLATE,
    EVAL_ANSWER_GEN_TEMPLATE,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ============ 配置项 ============
DATA_MD_PATH = BASE_DIR / 'data' / 'md'
OUTPUT_FILE  = BASE_DIR / 'data' / 'ragas_eval_data.jsonl'

# 模型选择：支持环境变量覆盖，保持默认量级
EVAL_MODEL   = os.getenv("EVAL_MODEL",   "Qwen/Qwen3-8B")             # QA 出题：8B 速度快、成本低
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "Qwen/Qwen2.5-32B-Instruct") # Answer 生成：32B 确保质量

MAX_CHUNKS_PER_FILE    = int(os.getenv("MAX_CHUNKS_PER_FILE", "5"))   # 单文件最大采样分块数
MAX_FILE_WORKERS       = int(os.getenv("MAX_FILE_WORKERS",    "3"))    # 文件级并发线程数
MAX_CONCURRENT_CALLS   = int(os.getenv("MAX_CONCURRENT_CALLS","8"))    # 全局 LLM 请求并发上限（信号量）
FALLBACK_CONCURRENT    = 3                                              # 熔断降级后的并发数
FAILURE_WINDOW_SECS    = 30                                             # 熔断检测窗口（秒）
FAILURE_THRESHOLD      = 5                                              # 窗口内失败次数触发熔断阈值
RETRY_TIMES            = 3                                              # 单次调用最大重试次数
RETRY_BASE_DELAY       = 2.0                                            # 指数退避基础等待秒数


# ============ 熔断器（并发失败监控 + 动态降级）============

class CircuitBreaker:
    """
    滑动窗口熔断器。
    在 FAILURE_WINDOW_SECS 内若失败次数 >= FAILURE_THRESHOLD，
    则将全局信号量从初始值降级至 FALLBACK_CONCURRENT，防止继续打爆 API。
    """

    def __init__(self, semaphore: threading.Semaphore):
        self._lock          = threading.Lock()
        self._failure_times: List[float] = []
        self._semaphore     = semaphore
        self._tripped       = False

    def record_failure(self):
        now = time.monotonic()
        with self._lock:
            # 清除窗口外的旧记录
            self._failure_times = [t for t in self._failure_times if now - t < FAILURE_WINDOW_SECS]
            self._failure_times.append(now)

            if not self._tripped and len(self._failure_times) >= FAILURE_THRESHOLD:
                self._trip()

    def _trip(self):
        """熔断触发：打印告警并重新补充信号量，使有效并发降为 FALLBACK_CONCURRENT。"""
        self._tripped = True
        current_available = self._semaphore._value          # 当前剩余许可数（内部属性）
        # 通过反复 acquire() 消耗多余许可，将上限降为 FALLBACK_CONCURRENT
        to_drain = current_available - FALLBACK_CONCURRENT
        for _ in range(max(0, to_drain)):
            # 非阻塞尝试消耗多余许可
            if not self._semaphore.acquire(blocking=False):
                break
        print(
            f"\n🔴 [熔断] {FAILURE_WINDOW_SECS}s 内失败 {len(self._failure_times)} 次，"
            f"并发上限已降级至 {FALLBACK_CONCURRENT}"
        )

    @property
    def tripped(self) -> bool:
        return self._tripped


# 全局并发信号量 & 熔断器（模块加载时初始化）
_global_semaphore    = threading.Semaphore(MAX_CONCURRENT_CALLS)
_circuit_breaker     = CircuitBreaker(_global_semaphore)

# 全局计数器（线程安全）
_stats_lock          = threading.Lock()
_api_fail_count      = 0
_json_fail_count     = 0


def _inc_api_fail():
    global _api_fail_count
    with _stats_lock:
        _api_fail_count += 1


def _inc_json_fail():
    global _json_fail_count
    with _stats_lock:
        _json_fail_count += 1


# ============ 模型调用（带重试 + 限流保护）============

def call_eval_model(llm_client: UnifiedLLMClient, prompt: str, model: str = None) -> str:
    """
    调用评测专用大模型，关闭思考模式 (Qwen3 no_think)。
    - 使用全局信号量控制并发，防止 API 被打爆。
    - 遭遇 429/500 等限流错误时，指数退避重试最多 RETRY_TIMES 次。
    - 连续失败上报熔断器，触发动态并发降级。
    """
    target_model = model or EVAL_MODEL
    messages = [
        {"role": "system", "content": "你是一个专业的 RAG 评测数据集生成专家。请严格按照用户要求的格式输出。"},
        {"role": "user",   "content": prompt},
    ]
    extra_body = {}
    if "Qwen3" in target_model or "qwen3" in target_model:
        extra_body["enable_thinking"] = False

    last_error = None
    for attempt in range(1, RETRY_TIMES + 1):
        with _global_semaphore:                        # 全局并发限制
            try:
                response = llm_client.client.chat.completions.create(
                    messages=messages,
                    model=target_model,
                    stream=False,
                    extra_body=extra_body if extra_body else None,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                err_str    = str(e)
                is_rate    = any(kw in err_str for kw in ("429", "rate", "Rate", "limit", "capacity"))
                is_server  = any(kw in err_str for kw in ("500", "502", "503", "timeout", "Timeout"))

                if is_rate or is_server:
                    _circuit_breaker.record_failure()
                    _inc_api_fail()
                    delay = RETRY_BASE_DELAY ** attempt
                    print(
                        f"\n  ⚠️ [{target_model}] 第 {attempt}/{RETRY_TIMES} 次调用失败（限流/服务异常），"
                        f"{delay:.0f}s 后重试... 原因: {err_str[:80]}"
                    )
                    time.sleep(delay)
                else:
                    # 非限流类错误：直接上报并退出重试
                    _circuit_breaker.record_failure()
                    _inc_api_fail()
                    print(f"\n  ❌ [{target_model}] 调用异常（非限流）: {err_str[:120]}")
                    return ""

    print(f"\n  ❌ [{target_model}] 已达最大重试次数，放弃。最后错误: {str(last_error)[:120]}")
    _inc_api_fail()
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
        splits   = splitter.split_text(content)

        if not splits:
            return [content[:1500]] if len(content) > 50 else []

        # 筛选长度合适的分块（200~1500 字符）
        valid_chunks = [
            chunk.page_content.strip()
            for chunk in splits
            if 200 < len(chunk.page_content.strip()) < 1500
        ]

        if not valid_chunks:
            longest = max(splits, key=lambda c: len(c.page_content))
            text    = longest.page_content.strip()
            return [text[:1500]] if len(text) > 50 else []

        # 均匀采样：超过上限时间隔取样
        if len(valid_chunks) > max_chunks:
            step         = len(valid_chunks) / max_chunks
            valid_chunks = [valid_chunks[int(i * step)] for i in range(max_chunks)]

        return valid_chunks
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return []


# ============ 文档摘要生成 ============

def generate_document_summary(llm_client: UnifiedLLMClient, file_path: Path) -> str:
    """生成文档的整体摘要，用于跨章节出题。截取前 4000 字符以避免溢出上下文。"""
    try:
        content   = file_path.read_text(encoding='utf-8')
        truncated = content[:4000]
        prompt    = EVAL_SUMMARY_TEMPLATE.format(content=truncated)
        return call_eval_model(llm_client, prompt)
    except Exception as e:
        print(f"生成摘要失败 {file_path}: {e}")
        return ""


# ============ JSON 解析（五级兜底策略）============

def _normalize_python_literals(text: str) -> str:
    """
    将模型输出的 Python 字面量规范化为合法 JSON：
    True → true, False → false, None → null
    仅替换独立单词，避免误替换字符串内容中的子串。
    """
    text = re.sub(r'(?<![\w"\'])\bTrue\b(?![\w"\'])',  'true',  text)
    text = re.sub(r'(?<![\w"\'])\bFalse\b(?![\w"\'])', 'false', text)
    text = re.sub(r'(?<![\w"\'])\bNone\b(?![\w"\'])',  'null',  text)
    return text


def _extract_json_array(text: str) -> Optional[str]:
    """
    使用括号深度追踪，从文本中提取第一个完整的 JSON 数组 [...]。
    比非贪婪正则更健壮：能正确处理 answer 中含 ']' 的情况（如代码片段、公式）。
    """
    start = text.find('[')
    if start == -1:
        return None
    depth   = 0
    in_str  = False
    escape  = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
    return None


def parse_qa_json(response: str) -> List[Dict[str, str]]:
    """
    从模型响应中解析 JSON 格式的 QA 对。五级解析策略：

    策略一：剥离 Markdown 代码块 + Python 字面量规范化后整体解析
    策略二：括号深度追踪精准提取 JSON 数组（修复旧非贪婪正则误截断问题）
    策略三：逐行扫描截取 JSON 数组边界后解析
    策略四：去除末尾逗号等常见错误后解析
    策略五：强制截断不完整 JSON，补全为合法结构后解析

    任何一级成功即返回，全部失败则打印诊断信息后返回 []。
    """
    if not response:
        return []

    def _validate(qa_list) -> List[Dict[str, str]]:
        """校验并提取有效的 QA 对。
        阈值降为 > 1，允许数字、短词等合法factual答案（如 '1024'、'平滑幂定律'）。
        """
        if not isinstance(qa_list, list):
            return []
        return [
            {"question": item["question"].strip(), "answer": item["answer"].strip()}
            for item in qa_list
            if isinstance(item, dict)
            and "question" in item and "answer" in item
            and len(item["question"].strip()) > 5   # 问题至少有意义
            and len(item["answer"].strip()) > 1     # ← 修复：从 >5 降至 >1，避免过滤短factual答案
        ]

    def _try_parse(text: str) -> List[Dict[str, str]]:
        try:
            return _validate(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            return []

    def _preprocess(text: str) -> str:
        """公共预处理：剥离代码块标记 + Python 字面量规范化"""
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = text.replace('```', '').strip()
        text = _normalize_python_literals(text)
        return text

    preprocessed = _preprocess(response)

    # ------ 策略一：预处理后整体解析 ------
    result = _try_parse(preprocessed)
    if result:
        return result

    # ------ 策略二：括号深度追踪精准提取（替代旧非贪婪正则）------
    # 旧方案 re.search(r'\[.*?\]') 在 answer 含 ']' 时会过早截断
    candidate = _extract_json_array(preprocessed)
    if candidate:
        result = _try_parse(candidate)
        if result:
            return result

    # ------ 策略三：逐行扫描，找 [ 到 ] 边界 ------
    lines     = preprocessed.splitlines()
    start_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith('[')), None)
    end_idx   = next((i for i, ln in enumerate(reversed(lines)) if ln.strip().endswith(']')), None)
    if start_idx is not None and end_idx is not None:
        end_real = len(lines) - 1 - end_idx
        if end_real >= start_idx:
            result = _try_parse("\n".join(lines[start_idx: end_real + 1]))
            if result:
                return result

    # ------ 策略四：修复末尾逗号等格式问题 ------
    fixed = re.sub(r',\s*([}\]])', r'\1', preprocessed)
    # 注意：不做全局单引号替换（会破坏含撇号的字符串），只修复键名单引号
    fixed = re.sub(r"(?<=[{,])\s*'([^']+)'\s*:", r' "\1":', fixed)
    result = _try_parse(fixed)
    if result:
        return result

    # ------ 策略五：截断修复不完整 JSON ------
    # 针对模型输出被截断的情况（如超出 max_tokens 或网络问题）
    # 找到最后一个完整的 {...} 对象，补全为合法数组
    obj_matches = list(re.finditer(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', preprocessed, re.S))
    if obj_matches:
        complete_objs = [m.group() for m in obj_matches]
        try:
            reconstructed = '[' + ','.join(complete_objs) + ']'
            result = _try_parse(reconstructed)
            if result:
                return result
        except Exception:
            pass

    # 全部失败，输出诊断信息
    _inc_json_fail()
    snippet = response[:200].replace('\n', '↵')
    print(f"  ⚠️ JSON 解析失败（五级策略均无效），响应片段: {snippet!r}")
    return []


# ============ Answer 生成 ============

def generate_answer_for_qa(llm_client: UnifiedLLMClient, question: str, chunk_content: str) -> str:
    """
    基于原始 chunk 片段，调用大模型生成该 QA 对的 answer 字段。
    answer 模拟线上 RAG 系统的回答，用于 faithfulness 评测。
    使用质量更高的 ANSWER_MODEL，并关闭 Qwen3 的思考模式。
    """
    if not question or not chunk_content:
        return ""
    prompt = EVAL_ANSWER_GEN_TEMPLATE.format(
        chunk_content=chunk_content,
        question=question,
    )
    return call_eval_model(llm_client, prompt, model=ANSWER_MODEL)


# ============ QA 生成 ============

def generate_chunk_qa(llm_client: UnifiedLLMClient, chunk_content: str, file_name: str) -> List[Dict[str, str]]:
    """基于单个 chunk 生成简单/中等难度的 QA 对"""
    if not chunk_content or len(chunk_content) < 50:
        return []
    prompt   = EVAL_CHUNK_QA_TEMPLATE.format(file_name=file_name, chunk_content=chunk_content)
    response = call_eval_model(llm_client, prompt)
    return parse_qa_json(response)


def generate_cross_chapter_qa(llm_client: UnifiedLLMClient, summary: str, file_name: str) -> List[Dict[str, str]]:
    """基于文档摘要生成跨章节的较难 QA 对"""
    if not summary or len(summary) < 100:
        return []
    prompt   = EVAL_CROSS_CHAPTER_QA_TEMPLATE.format(file_name=file_name, summary=summary)
    response = call_eval_model(llm_client, prompt)
    return parse_qa_json(response)


# ============ 单文件处理流水线 ============

def process_file(file_path: Path, llm_client: UnifiedLLMClient) -> List[Dict]:
    """
    处理单个文件，返回多层次的评测数据项。
    第一层：基于单个 chunk 的简单题（difficulty=simple）
      - chunk-level QA 生成与 Answer 生成在文件内并行执行（受全局信号量约束）
    第二层：基于文档摘要的跨章节难题（difficulty=cross_chapter）

    每条记录包含完整的四维度字段：
      question     - 问题
      contexts     - 召回片段（chunk 原文）
      answer       - 模型基于 contexts 生成的 RAG 回答（用于 faithfulness 评测）
      ground_truth - 标准答案
    """
    rel_path      = file_path.relative_to(DATA_MD_PATH)
    file_name     = file_path.name
    source_domain = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
    final_items   = []

    # ===== 第一层：基于单个 chunk 的简单题 =====
    chunks = extract_chunks(file_path)

    def _process_chunk(args):
        """在线程内处理单个 chunk：生成 QA 对并逐条生成 Answer。"""
        idx, chunk = args
        qa_results = generate_chunk_qa(llm_client, chunk, file_name)
        items      = []
        for qa in qa_results:
            question     = qa["question"]
            ground_truth = qa["answer"]
            answer       = generate_answer_for_qa(llm_client, question, chunk)
            items.append({
                "question":     question,
                "contexts":     [chunk],
                "answer":       answer,
                "ground_truth": ground_truth,
                "metadata": {
                    "source_file":   file_name,
                    "full_path":     str(rel_path),
                    "source_domain": source_domain,
                    "difficulty":    "simple",
                    "chunk_index":   idx,
                },
            })
        return items

    # chunk 内部并发：受全局信号量约束，不需额外限流
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), MAX_FILE_WORKERS)) as chunk_executor:
        chunk_futures = [chunk_executor.submit(_process_chunk, (i, c)) for i, c in enumerate(chunks)]
        for fut in concurrent.futures.as_completed(chunk_futures):
            try:
                final_items.extend(fut.result())
            except Exception as e:
                print(f"\n  ❌ chunk 处理出错 [{file_name}]: {e}")

    # ===== 第二层：基于文档摘要的跨章节难题 =====
    summary = generate_document_summary(llm_client, file_path)
    if summary:
        cross_qa = generate_cross_chapter_qa(llm_client, summary, file_name)
        for qa in cross_qa:
            question     = qa["question"]
            ground_truth = qa["answer"]
            answer       = generate_answer_for_qa(llm_client, question, summary)
            final_items.append({
                "question":     question,
                "contexts":     [summary],
                "answer":       answer,
                "ground_truth": ground_truth,
                "metadata": {
                    "source_file":   file_name,
                    "full_path":     str(rel_path),
                    "source_domain": source_domain,
                    "difficulty":    "cross_chapter",
                },
            })

    return final_items


# ============ 断点续跑：加载已处理文件集合 ============

def load_processed_files(output_file: Path) -> Set[str]:
    """
    从已有的输出 JSONL 中读取已处理的 source_file 列表，
    用于跳过重复处理，支持断点续跑。
    """
    processed: Set[str] = set()
    if not output_file.exists():
        return processed
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    src  = item.get("metadata", {}).get("source_file", "")
                    if src:
                        processed.add(src)
                except json.JSONDecodeError:
                    continue
        if processed:
            print(f"📌 断点续跑：检测到已有 {len(processed)} 个文件的输出，将跳过这些文件。")
    except Exception as e:
        print(f"⚠️ 读取断点缓存失败（将全量重跑）: {e}")
    return processed


# ============ 主入口 ============

def main():
    global _api_fail_count, _json_fail_count

    print("🚀 启动评测数据集生成流程...")
    print(f"📌 QA 生成模型:          {EVAL_MODEL}")
    print(f"📌 Answer 生成模型:      {ANSWER_MODEL}")
    print(f"📌 单文件最大采样分块数:  {MAX_CHUNKS_PER_FILE}")
    print(f"📌 文件级并发线程数:      {MAX_FILE_WORKERS}")
    print(f"📌 全局 LLM 并发上限:    {MAX_CONCURRENT_CALLS}（失败熔断后降至 {FALLBACK_CONCURRENT}）")
    print(f"📌 输出字段: question / contexts / answer / ground_truth\n")

    llm_client = UnifiedLLMClient()

    # 搜集文件 & 断点续跑
    all_files       = list(DATA_MD_PATH.rglob('*.md'))
    processed_files = load_processed_files(OUTPUT_FILE)
    pending_files   = [f for f in all_files if f.name not in processed_files]

    total_files    = len(all_files)
    skipped_count  = total_files - len(pending_files)
    print(f"📂 发现 {total_files} 个 Markdown 文件，跳过已处理 {skipped_count} 个，本次处理 {len(pending_files)} 个。\n")

    if not pending_files:
        print("✅ 所有文件已处理完毕，无需重跑。")
        return

    processed_count   = skipped_count
    total_qa_count    = 0
    simple_count      = 0
    cross_count       = 0
    answer_miss_count = 0

    # 追加模式写入（支持断点续跑）
    write_mode = 'a' if processed_files else 'w'
    with open(OUTPUT_FILE, write_mode, encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_FILE_WORKERS) as executor:
            future_to_file = {executor.submit(process_file, f, llm_client): f for f in pending_files}

            for future in concurrent.futures.as_completed(future_to_file):
                processed_count += 1
                source_file      = future_to_file[future]
                try:
                    items = future.result()
                    if items:
                        for item in items:
                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            total_qa_count += 1
                            difficulty      = item["metadata"]["difficulty"]
                            if difficulty == "simple":
                                simple_count += 1
                            else:
                                cross_count += 1
                            if not item.get("answer"):
                                answer_miss_count += 1
                        f_out.flush()

                    cb_status = "🔴已熔断" if _circuit_breaker.tripped else "🟢正常"
                    print(
                        f"进度: [{processed_count}/{total_files}] "
                        f"| QA: {total_qa_count} (简单:{simple_count} 跨章:{cross_count}) "
                        f"| API失败:{_api_fail_count} JSON失败:{_json_fail_count} "
                        f"| 熔断:{cb_status} "
                        f"| {source_file.name}",
                        end='\r',
                    )
                except Exception as e:
                    print(f"\n❌ 处理出错 [{source_file.name}]: {e}")

    print(f"\n\n✨ 任务完成！")
    print(f"📊 统计结果：")
    print(f"   处理文件总数:            {total_files} 个（本次新处理 {processed_count - skipped_count} 个）")
    print(f"   - 简单题 (chunk-level):  {simple_count} 条")
    print(f"   - 跨章节 (cross-chap):   {cross_count} 条")
    print(f"   - 总计:                  {total_qa_count} 条")
    if answer_miss_count > 0:
        print(f"   ⚠️  answer 字段为空:      {answer_miss_count} 条")
    if _api_fail_count > 0:
        print(f"   ⚠️  API 调用失败次数:     {_api_fail_count} 次")
    if _json_fail_count > 0:
        print(f"   ⚠️  JSON 解析失败次数:    {_json_fail_count} 次")
    if _circuit_breaker.tripped:
        print(f"   🔴 并发熔断已触发（降级至 {FALLBACK_CONCURRENT}），建议检查 API 限速配置")
    print(f"💾 结果保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
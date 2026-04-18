import sys
import os
import time
import concurrent.futures
from typing import List
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm_client import UnifiedLLMClient
import src.core.template as template

# 初始化测试所需的通用模型客户端
llm_client = UnifiedLLMClient()

def determine_retrieval_level(query: str) -> int:
    system_prompt = template.NEED_ENHANCEMENT_JUDGE_TEMPLATE.replace("{query}", query)
    result = llm_client.call_small_model(system_prompt=system_prompt).strip()
    if "2" in result: return 2
    elif "1" in result: return 1
    return 0

def extract_keywords_and_resolve(query: str) -> str:
    system_prompt = template.SINGLE_REWRITE_TEMPLATE.replace("{query}", query)
    return llm_client.call_small_model(system_prompt=system_prompt)

def rewrite_query(query: str) -> List[str]:
    system_prompt = template.MULTI_QUERY_REWRITE_TEMPLATE.replace("{query}", query)
    content = llm_client.call_small_model(system_prompt=system_prompt)
    try:
        if "```json" in content: content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content: content = content.split("```")[-1].split("```")[0].strip()
        result = json.loads(content)
        if isinstance(result, list): return result
    except:
        pass
    return []

def generate_hyde_document(query: str) -> str:
    system_prompt = template.HYDE_TEMPLATE_XINGHUO.replace("{query}", query)
    return llm_client.call_small_model(system_prompt=system_prompt)


def run_latency_test(query: str, desc: str):
    print("=" * 60)
    print(f"[{desc}]\n问题内容: {query}")
    print("-" * 60)
    
    start_time = time.perf_counter()
    
    # 获取等级判定时间
    level = determine_retrieval_level(query)
    level_time = time.perf_counter()
    print(f"✅ RAG 路由分级完成 => 触发 Level {level} (判定耗时: {(level_time - start_time)*1000:.2f} ms)")
    
    if level == 0:
        print("   -> (无需改写) 直接拿着问题去匹配知识库。")
    
    elif level == 1:
        refined = extract_keywords_and_resolve(query)
        action_time = time.perf_counter()
        print(f"✅ Level 1 短效提炼完成 (提炼耗时: {(action_time - level_time)*1000:.2f} ms)")
        print(f"   -> 原问题被精简为: '{refined}'")
        
    elif level == 2:
        print("   -> 正在并发请求多组重写子词与 HyDE 参考文档...")
        # 利用多线程测试 API 请求并发现时
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_rewrite = executor.submit(rewrite_query, query)
            future_hyde = executor.submit(generate_hyde_document, query)
            
            sub_queries = future_rewrite.result()
            hyde_doc = future_hyde.result()
            
        action_time = time.perf_counter()
        print(f"✅ Level 2 全量处理完成 (并发处理总耗时: {(action_time - level_time)*1000:.2f} ms)")
        print(f"   -> 多维查询扩展: {sub_queries}")
        print(f"   -> 假想文档生成(截取前50词): {hyde_doc[:50]}...")
        
    total_time = time.perf_counter()
    print("-" * 60)
    print(f"🔥 Retrieval 知识收集前置模块总计耗时: {(total_time - start_time)*1000:.2f} 毫秒")


if __name__ == "__main__":
    test_cases = [
        ("什么是校招？", "测试 Level 0 (Zero-shot) 简单信息流"),
        ("它和算法岗比起来薪资怎么样？需要会啥？", "测试 Level 1 (Single Rewrite) 模糊单点追踪"),
        ("没有好的落地项目，我是应该冲互联网的测开，还是硬切底层嵌入式？各自前景和所需技能点差异在哪？", "测试 Level 2 (HyDE/Multi-step) 深层专家解析向")
    ]
    
    print("\n🚀 开始三级 RAG 延迟性能评估 (基于 Qwen2.5-7B 的快速路由)\n")
    
    for q, desc in test_cases:
        run_latency_test(q, desc)
        time.sleep(1) # 请求防抖
    
    print("\n✅ 所有测速靶场跑测完毕。\n")

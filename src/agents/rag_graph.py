"""
LangGraph 自纠正 RAG 反思图 (Self-Corrective Agentic RAG Graph)

对齐开发规范 plan.md §4:
- §4.1: AgentState 定义（所有节点通过 State 交换数据，禁止全局变量）
- §4.2: 节点原子性（每个 Node 只负责一个动作）
- §3.2: 命名约定（node_ / decide_ 前缀）

流程:
  node_retrieve -> node_grade_docs -> [decide_is_relevant]
    -> 相关:    node_generate -> node_reflect -> [decide_is_faithful]
                  -> 通过:   END
                  -> 未通过:  node_rewrite -> node_retrieve (循环)
    -> 不相关:  node_rewrite -> node_retrieve (循环)
"""

import json
from typing import TypedDict, List, Optional, Any

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from src.core.llm_client import UnifiedLLMClient
from src.eval.Recall_test import execute_retrieval_pipeline
import src.core.template as template

# ============================================================
# 全局配置
# ============================================================
MAX_REFLECTION_STEPS = 2          # 最大反思轮数（防环死）
FAITHFUL_SCORE_THRESHOLD = 0.8    # 忠实度阈值
INTERNAL_GENERATE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 图内部验证用模型（快速便宜）

# ============================================================
# 模块级单例
# ============================================================
_llm_client: Optional[UnifiedLLMClient] = None


def _get_llm_client() -> UnifiedLLMClient:
    """获取模块级 LLM 客户端单例（与 Recall_test.py 做法一致）"""
    global _llm_client
    if _llm_client is None:
        _llm_client = UnifiedLLMClient()
    return _llm_client


# ============================================================
# 1. AgentState 定义 (对齐 plan.md §4.1)
# ============================================================
class AgentState(TypedDict):
    query: str                              # 原始问题
    rewritten_query: str                    # 改写后的搜索词
    context: List[Document]                 # 召回片段
    generation: str                         # 模型生成内容
    reflection_log: List[str]               # 审计日志
    steps_count: int                        # 计数器（防环死）
    faithful_score: float                   # 忠实度评分
    # --- 外部注入（不随图变化） ---
    profile_text: str
    profile_vec: Optional[List[float]]
    profile_filter: Optional[dict]
    retrieval_status: str                   # 检索状态信息


# ============================================================
# 2. 原子节点函数 (对齐 plan.md §4.2)
# ============================================================

def node_retrieve(state: AgentState) -> dict:
    """
    混合检索节点 — 调用 Recall_test.execute_retrieval_pipeline
    
    使用 rewritten_query（若非空）或 query 作为检索输入。
    """
    llm_client = _get_llm_client()
    search_query = state.get("rewritten_query") or state["query"]

    retrieval_result = execute_retrieval_pipeline(
        query=search_query,
        llm_client=llm_client,
        profile_text=state.get("profile_text", ""),
        profile_vec=state.get("profile_vec"),
        profile_filter=state.get("profile_filter"),
    )

    return {
        "context": retrieval_result["final_docs"],
        "retrieval_status": retrieval_result["status_message"],
    }


def node_grade_docs(state: AgentState) -> dict:
    """
    文档评分节点 — 使用小模型逐篇判断文档与 query 的相关性。
    
    过滤掉不相关文档，只保留被判定为 "yes" 的文档。
    """
    llm_client = _get_llm_client()
    query = state["query"]
    docs = state["context"]

    if not docs:
        return {"context": []}

    relevant_docs = []
    for doc in docs:
        prompt = template.DOC_GRADING_TEMPLATE.format(
            query=query,
            document=doc.page_content[:500]  # 截断避免超长
        )
        try:
            result = llm_client.call_small_model(system_prompt=prompt).strip().lower()
            if "yes" in result:
                relevant_docs.append(doc)
        except Exception as e:
            print(f"[node_grade_docs] 评分异常: {e}")
            # 评分失败的文档保守保留
            relevant_docs.append(doc)

    return {"context": relevant_docs}


def node_rewrite(state: AgentState) -> dict:
    """
    查询改写节点 — 基于反思日志改写 query。
    
    如果 reflection_log 非空，使用 REFLECTION_REWRITE_TEMPLATE；
    否则回退到 SINGLE_REWRITE_TEMPLATE。
    同时递增 steps_count。
    """
    llm_client = _get_llm_client()
    query = state["query"]
    reflection_log = state.get("reflection_log", [])
    steps = state.get("steps_count", 0)

    if reflection_log:
        prompt = template.REFLECTION_REWRITE_TEMPLATE.format(
            query=query,
            reflection_log="\n".join(reflection_log)
        )
    else:
        prompt = template.SINGLE_REWRITE_TEMPLATE.replace("{query}", query)

    try:
        rewritten = llm_client.call_small_model(system_prompt=prompt).strip()
    except Exception as e:
        print(f"[node_rewrite] 改写异常: {e}")
        rewritten = query  # 改写失败则保留原查询

    return {
        "rewritten_query": rewritten,
        "steps_count": steps + 1,
    }


def node_generate(state: AgentState) -> dict:
    """
    大模型生成节点 — 非流式调用，返回 generation。
    
    使用内部快速模型进行"验证用"生成，为后续反思节点提供素材。
    最终面向用户的流式输出由 main.py 单独完成（保证体验）。
    """
    llm_client = _get_llm_client()
    query = state["query"]
    docs = state["context"]

    # 构建上下文
    context_text = "\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    # 使用系统提示词模板
    system_prompt = template.RAG_TEMPLATE_XINGHUO.replace("{context}", context_text).replace("{query}", query)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = llm_client.call_large_model(
            messages=messages,
            model_name=INTERNAL_GENERATE_MODEL,
            stream=False,
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[node_generate] 生成异常: {e}")
        content = ""

    return {"generation": content}


import re
import json

def node_reflect(state: AgentState) -> dict:
    """
    反思节点 — 评估 faithful_score。
    """
    llm_client = _get_llm_client()
    query = state["query"]
    docs = state["context"]
    generation = state["generation"]
    reflection_log = list(state.get("reflection_log", []))

    context_text = "\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    prompt = template.REFLECTION_TEMPLATE.format(
        query=query,
        context=context_text,
        generation=generation,
    )

    faithful_score = 0.0
    try:
        raw_text = llm_client.call_small_model(system_prompt=prompt).strip()
        
        # --- 核心修复逻辑：使用正则提取第一个匹配的 JSON 对象 ---
        # re.DOTALL 允许 . 匹配换行符
        json_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
        
        if json_match:
            result_text = json_match.group(1)
        else:
            # 如果正则没匹配到，尝试最后的倔强：清理掉 markdown 标签
            result_text = raw_text.replace("```json", "").replace("```", "").strip()

        # 执行解析
        result = json.loads(result_text)
        
        # 提取字段
        faithful_score = float(result.get("faithful_score", 0.0))
        issues = result.get("issues", [])
        suggestion = result.get("suggestion", "")

        # 追加反思日志
        log_entry = f"[轮次{len(reflection_log)+1}] 忠实度={faithful_score:.2f}"
        if issues:
            log_entry += f" 问题: {'; '.join(issues)}"
        if suggestion:
            log_entry += f" 建议: {suggestion}"
        reflection_log.append(log_entry)

    except Exception as e:
        # 打印原始文本方便调试
        print(f"[node_reflect] 解析失败。原始文本: {raw_text[:200]}...")
        print(f"[node_reflect] 异常详情: {e}")
        
        faithful_score = 1.0  # 容错处理
        reflection_log.append(f"[轮次{len(reflection_log)+1}] 反思评估异常，默认通过: {e}")

    return {
        "faithful_score": faithful_score,
        "reflection_log": reflection_log,
    }


# ============================================================
# 3. 条件边路由函数 (对齐 plan.md §3.2 decide_ 前缀)
# ============================================================

def decide_is_relevant(state: AgentState) -> str:
    """
    根据 node_grade_docs 的过滤结果路由：
    - 如果过滤后仍有文档 -> "relevant" -> node_generate
    - 如果文档全被过滤掉 -> "not_relevant" -> node_rewrite
    
    但如果已经达到最大反思轮数，即使不相关也强制进入生成（防环死）。
    """
    docs = state.get("context", [])
    steps = state.get("steps_count", 0)

    if docs:
        return "relevant"
    elif steps >= MAX_REFLECTION_STEPS:
        # 已经重写多轮仍无结果，强制进入生成（防环死）
        return "relevant"
    else:
        return "not_relevant"


def decide_is_faithful(state: AgentState) -> str:
    """
    根据 faithful_score 和 steps_count 路由：
    - faithful_score >= 阈值 -> "faithful" -> END
    - 未达标且未超限 -> "not_faithful" -> node_rewrite
    - 已达最大轮数 -> "max_steps" -> END（强制返回当前结果）
    """
    score = state.get("faithful_score", 0.0)
    steps = state.get("steps_count", 0)

    if score >= FAITHFUL_SCORE_THRESHOLD:
        return "faithful"
    elif steps >= MAX_REFLECTION_STEPS:
        print(f"[decide_is_faithful] 达到最大反思轮数 {MAX_REFLECTION_STEPS}，强制返回")
        return "max_steps"
    else:
        return "not_faithful"


# ============================================================
# 4. 编译图
# ============================================================

def build_rag_graph():
    """
    构建并编译自纠正 RAG 反思图。
    
    图结构:
      node_retrieve -> node_grade_docs -> [decide_is_relevant]
        -> relevant:     node_generate -> node_reflect -> [decide_is_faithful]
                           -> faithful:     END
                           -> not_faithful: node_rewrite -> node_retrieve (循环)
                           -> max_steps:    END
        -> not_relevant: node_rewrite -> node_retrieve (循环)
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("node_retrieve", node_retrieve)
    graph.add_node("node_grade_docs", node_grade_docs)
    graph.add_node("node_rewrite", node_rewrite)
    graph.add_node("node_generate", node_generate)
    graph.add_node("node_reflect", node_reflect)

    # 设置入口
    graph.set_entry_point("node_retrieve")

    # 固定边
    graph.add_edge("node_retrieve", "node_grade_docs")
    graph.add_edge("node_rewrite", "node_retrieve")
    graph.add_edge("node_generate", "node_reflect")

    # 条件边
    graph.add_conditional_edges(
        "node_grade_docs",
        decide_is_relevant,
        {"relevant": "node_generate", "not_relevant": "node_rewrite"},
    )
    graph.add_conditional_edges(
        "node_reflect",
        decide_is_faithful,
        {"faithful": END, "not_faithful": "node_rewrite", "max_steps": END},
    )

    return graph.compile()

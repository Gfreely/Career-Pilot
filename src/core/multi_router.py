"""
multi_router.py — 多路由并行意图分发器

核心职责:
  1. analyse_query()     — 单次 LLM 调用，同时完成:
       • 多标签意图识别  (intents: List[str])
       • 查询改写优化    (rewritten_query: str)
       • 实体提取        (entities: dict)

  2. MultiRouteDispatcher — 根据 intents 并行激活各路由，合并结果:
       • RAG         → LangGraph 自纠正 RAG 图
       • MCP_JOB     → mcp_stub.execute_mcp_job_retrieval
       • MCP_COMPANY → mcp_stub.execute_mcp_company_insight
       • DIRECT      → 跳过检索，直接返回空 context

路由标签:
  RAG | MCP_JOB | MCP_COMPANY | DIRECT
  DIRECT 与其余互斥; 其余可多选并行执行。

并发策略:
  使用 concurrent.futures.ThreadPoolExecutor。
  MCP 路由超时阈值: MCP_TIMEOUT_SECONDS (默认 5 秒)。
  LangGraph RAG 图阻塞执行（无超时），MCP 超时后降级为空 context。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, Dict, List

from src.utils.json_utils import parse_json_object
import src.core.template as template
from src.core.mcp_stub import execute_mcp_job_retrieval, execute_mcp_company_insight

if TYPE_CHECKING:
    from src.core.llm_client import UnifiedLLMClient


# ============================================================
# 全局配置
# ============================================================
MCP_TIMEOUT_SECONDS: float = 5.0   # MCP 路由超时（秒）；超时后降级为空
VALID_INTENTS = {"RAG", "MCP_JOB", "MCP_COMPANY", "DIRECT"}
DIRECT_FALLBACK_PATTERNS = (
    r"^\s*(你好|您好|hi|hello|早上好|中午好|晚上好)[！!，,。.\s]*$",
    r"^\s*(谢谢|多谢|thx|thanks)[！!，,。.\s]*$",
    r"^\s*(你是谁|介绍一下你自己|你能做什么)[？?！!\s]*$",
)


def _build_default_result(query: str) -> Dict[str, Any]:
    return {
        "intents": ["RAG"],
        "rewritten_query": query,
        "entities": {"company": None, "position": None, "location": None, "keywords": []},
        "confidence": 0.5,
        "reasoning": "解析失败，回退至默认 RAG 路由",
    }


def _build_heuristic_fallback(query: str) -> Dict[str, Any]:
    normalized_query = (query or "").strip()
    for pattern in DIRECT_FALLBACK_PATTERNS:
        if re.match(pattern, normalized_query, re.IGNORECASE):
            return {
                "intents": ["DIRECT"],
                "rewritten_query": "",
                "entities": {"company": None, "position": None, "location": None, "keywords": []},
                "confidence": 0.3,
                "reasoning": "模型 JSON 解析失败，按基础问候规则回退为 DIRECT",
            }
    return _build_default_result(normalized_query)


def _normalize_analysis_result(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    raw_intents = result.get("intents", ["RAG"])
    if isinstance(raw_intents, str):
        raw_intents = [raw_intents]
    elif not isinstance(raw_intents, list):
        raw_intents = ["RAG"]

    intents = [intent for intent in raw_intents if intent in VALID_INTENTS]
    if not intents:
        intents = ["RAG"]
    if "DIRECT" in intents and len(intents) > 1:
        intents = ["DIRECT"]

    entities = result.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}

    keywords = entities.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]
    elif not isinstance(keywords, list):
        keywords = []

    rewritten_query = str(result.get("rewritten_query", query) or query).strip()
    if intents == ["DIRECT"]:
        rewritten_query = ""

    confidence = result.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "intents": intents,
        "rewritten_query": rewritten_query,
        "entities": {
            "company": entities.get("company"),
            "position": entities.get("position"),
            "location": entities.get("location"),
            "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()],
        },
        "confidence": confidence,
        "reasoning": str(result.get("reasoning", "")).strip(),
    }


def _clean_scalar_fragment(value: str) -> str:
    """清洗模型返回的单字段片段，尽量保留语义，移除结构噪音。"""
    cleaned = (value or "").strip().strip(",")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace('\\"', '"')
    return " ".join(cleaned.split())


def _extract_optional_string(text: str, field_name: str) -> Any:
    pattern = rf'"{field_name}"\s*[:：]?\s*(null|"[^"\r\n]*"|.+?)(?=\s*,?\s*"[A-Za-z_]+"\s*[:：]?\s*|\s*,?\s*\}}|\s*$)'
    match = re.search(pattern, text, re.S)
    if not match:
        return None

    raw_value = match.group(1).strip()
    if raw_value.lower() == "null":
        return None

    if raw_value.startswith('"'):
        closing_quote = raw_value.rfind('"')
        if closing_quote > 0:
            raw_value = raw_value[: closing_quote + 1]

    value = _clean_scalar_fragment(raw_value)
    return value or None


def _salvage_analysis_result(raw_text: str, query: str) -> Dict[str, Any] | None:
    """当严格 JSON 恢复失败时，按 analyse_query 固定 schema 逐字段抢救。"""
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    intents = [intent for intent in VALID_INTENTS if re.search(rf'["\']?{intent}["\']?', cleaned, re.IGNORECASE)]
    if not intents:
        intents = ["RAG"]
    if "DIRECT" in intents and len(intents) > 1:
        intents = ["DIRECT"]

    rewritten_query = query
    rewritten_match = re.search(
        r'"rewritten_query"\s*[:：]?\s*(.+?)(?=\s*,?\s*"[A-Za-z_]+"\s*[:：]?\s*|\s*,?\s*\}}|\s*$)',
        cleaned,
        re.S,
    )
    if rewritten_match:
        rewritten_query = _clean_scalar_fragment(rewritten_match.group(1)) or query
    if intents == ["DIRECT"]:
        rewritten_query = ""

    reasoning = ""
    reasoning_match = re.search(
        r'"reasoning"\s*[:：]?\s*(.+?)(?=\s*,?\s*"[A-Za-z_]+"\s*[:：]?\s*|\s*,?\s*\}}|\s*$)',
        cleaned,
        re.S,
    )
    if reasoning_match:
        reasoning = _clean_scalar_fragment(reasoning_match.group(1))

    confidence = 0.5
    confidence_match = re.search(r'"confidence"\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)', cleaned)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
        except ValueError:
            confidence = 0.5

    keywords: List[str] = []
    keywords_match = re.search(r'"keywords"\s*[:：]?\s*(\[.*?\]|"[^"]+")', cleaned, re.S)
    if keywords_match:
        val = keywords_match.group(1)
        if val.startswith('['):
            for keyword in re.findall(r'"([^"\r\n]+)"', val):
                keyword = keyword.strip()
                if keyword:
                    keywords.append(keyword)
        else:
            keyword = val.strip('"').strip()
            if keyword:
                keywords.append(keyword)

    return {
        "intents": intents,
        "rewritten_query": rewritten_query,
        "entities": {
            "company": _extract_optional_string(cleaned, "company"),
            "position": _extract_optional_string(cleaned, "position"),
            "location": _extract_optional_string(cleaned, "location"),
            "keywords": keywords,
        },
        "confidence": confidence,
        "reasoning": reasoning or "模型输出非标准 JSON，已按字段恢复",
    }


# ============================================================
# 1. analyse_query — 意图识别 + 查询改写 (单次调用)
# ============================================================

def analyse_query(query: str, llm_client: "UnifiedLLMClient") -> dict:
    """
    一次 LLM 调用，同时返回多标签意图 + 改写查询 + 实体。

    Parameters
    ----------
    query : str
        用户原始输入。
    llm_client : UnifiedLLMClient
        统一 LLM 客户端（使用小模型，速度快成本低）。

    Returns
    -------
    dict
        {
            "intents":          List[str],   # e.g. ["RAG", "MCP_JOB"]
            "rewritten_query":  str,          # 改写后的检索词
            "entities":         dict,         # company/position/location/keywords
            "confidence":       float,
            "reasoning":        str,          # 调试用
        }

    Fallback
    --------
    解析失败时返回默认值: intents=["RAG"], rewritten_query=query
    """
    prompt = template.MULTI_ROUTE_ANALYSIS_TEMPLATE.replace("{{USER_QUERY}}", query)

    try:
        raw = llm_client.call_small_model(system_prompt=prompt).strip()
        try:
            result = parse_json_object(raw)
        except Exception:
            salvaged_result = _salvage_analysis_result(raw, query)
            if salvaged_result is not None:
                return _normalize_analysis_result(salvaged_result, query)
            raise
        return _normalize_analysis_result(result, query)

    except Exception as e:
        preview = raw[:200].replace("\n", "\\n") if "raw" in locals() else ""
        print(f"[analyse_query] 解析异常: {e}; 原始输出片段: {preview}")
        return _build_heuristic_fallback(query)


# ============================================================
# 2. MultiRouteDispatcher — 并行分发 & 合并
# ============================================================

class MultiRouteDispatcher:
    """
    根据 analyse_query 的 intents 列表并行执行各路由，合并 context。

    Usage
    -----
    dispatcher = MultiRouteDispatcher()
    result = dispatcher.dispatch(analysis, rag_graph, conversation_manager, emb_model)
    merged_context = result["merged_context"]
    """

    def dispatch(
        self,
        analysis: dict,
        rag_graph,                            # LangGraph CompiledGraph
        conversation_manager,                 # MemoryManager
        emb_model,                            # LocalBGEM3Embeddings
    ) -> dict:
        """
        并行执行所有激活的路由，合并返回。

        Parameters
        ----------
        analysis : dict
            由 analyse_query() 返回的分析结果。
        rag_graph : CompiledGraph
            已编译的 LangGraph RAG 图实例（单例）。
        conversation_manager : MemoryManager
            当前会话的内存管理器，提供 profile/vector/filter。
        emb_model : LocalBGEM3Embeddings
            用于计算画像向量的嵌入模型实例。

        Returns
        -------
        dict
            {
                "merged_context":  str,   # 所有路由 context 拼接（直接注入 system prompt）
                "route_results":   dict,  # 各路由原始结果（调试用）
                "active_routes":   list,  # 最终激活的路由列表
                "display_info":    str,   # 在 thinking_box 显示的状态文字
                "rag_final_state": dict,  # LangGraph 最终状态（含反思日志等，可能为空）
            }
        """
        intents         = analysis.get("intents", ["RAG"])
        rewritten_query = analysis.get("rewritten_query", "")
        entities        = analysis.get("entities", {})

        route_results:    Dict[str, dict] = {}
        rag_final_state:  dict            = {}

        # ── DIRECT 分支：跳过所有检索 ──
        if "DIRECT" in intents:
            return {
                "merged_context":  "",
                "route_results":   {"DIRECT": {"context": "", "status": "ok", "source": "DIRECT"}},
                "active_routes":   ["DIRECT"],
                "display_info":    "**判定意图：** 直接回答",
                "rag_final_state": {},
            }

        # ── 多路由并行执行 ──
        futures = {}
        with ThreadPoolExecutor(max_workers=len(intents)) as executor:

            for intent in intents:
                if intent == "RAG":
                    futures["RAG"] = executor.submit(
                        self._run_rag,
                        rewritten_query or entities.get("keywords", []) and " ".join(entities["keywords"]) or "",
                        rag_graph,
                        conversation_manager,
                        emb_model,
                    )
                elif intent == "MCP_JOB":
                    futures["MCP_JOB"] = executor.submit(
                        execute_mcp_job_retrieval,
                        entities,
                        rewritten_query,
                        MCP_TIMEOUT_SECONDS,
                    )
                elif intent == "MCP_COMPANY":
                    futures["MCP_COMPANY"] = executor.submit(
                        execute_mcp_company_insight,
                        entities,
                        rewritten_query,
                        MCP_TIMEOUT_SECONDS,
                    )

            # 收集结果（各路由独立超时）
            for route, future in futures.items():
                try:
                    if route == "RAG":
                        # RAG 不设外部超时（LangGraph 内部有防环死逻辑）
                        result = future.result()
                    else:
                        # MCP 路由超时降级
                        result = future.result(timeout=MCP_TIMEOUT_SECONDS)
                    route_results[route] = result

                    if route == "RAG":
                        rag_final_state = result.get("rag_final_state", {})

                except FuturesTimeoutError:
                    print(f"[MultiRouteDispatcher] {route} 超时 ({MCP_TIMEOUT_SECONDS}s)，已降级")
                    route_results[route] = {
                        "context": f"⚠️ [{route}] 数据获取超时，已跳过此路由。",
                        "status":  "timeout",
                        "source":  route,
                    }
                except Exception as e:
                    print(f"[MultiRouteDispatcher] {route} 执行异常: {e}")
                    route_results[route] = {
                        "context": f"⚠️ [{route}] 执行异常: {e}",
                        "status":  "error",
                        "source":  route,
                    }

        # ── 合并 context ──
        merged_parts = []
        for route in intents:
            res = route_results.get(route, {})
            ctx = res.get("context", "")
            if ctx:
                merged_parts.append(f"=== 来源: {route} ===\n{ctx}")

        merged_context = "\n\n".join(merged_parts)

        # ── 生成 display_info ──
        display_info = self._build_display_info(intents, route_results, rag_final_state)

        return {
            "merged_context":  merged_context,
            "route_results":   route_results,
            "active_routes":   intents,
            "display_info":    display_info,
            "rag_final_state": rag_final_state,
        }

    # ----------------------------------------------------------------
    # 内部 RAG 执行器
    # ----------------------------------------------------------------

    def _run_rag(
        self,
        search_query: str,
        rag_graph,
        conversation_manager,
        emb_model,
    ) -> dict:
        """
        执行 LangGraph RAG 图，返回结构化结果。

        Returns
        -------
        dict
            {
                "context":         str,   # 格式化的文档片段文本
                "status":          str,
                "source":          str,
                "rag_final_state": dict,  # LangGraph 最终状态
            }
        """
        from src.agents.rag_graph import AgentState

        profile_text   = conversation_manager.get_profile_text()
        profile_filter = conversation_manager.get_profile_filter()
        profile_vec    = conversation_manager.get_profile_vector(emb_model)

        initial_state: AgentState = {
            "query":            search_query,
            "rewritten_query":  "",
            "context":          [],
            "generation":       "",
            "reflection_log":   [],
            "steps_count":      0,
            "faithful_score":   0.0,
            "profile_text":     profile_text,
            "profile_vec":      profile_vec,
            "profile_filter":   profile_filter,
            "retrieval_status": "",
        }

        final_state = rag_graph.invoke(initial_state)

        context_text = "\n\n".join(
            [f"[{i+1}] {d.page_content}" for i, d in enumerate(final_state["context"])]
        )

        return {
            "context":         context_text,
            "status":          "ok",
            "source":          "RAG",
            "rag_final_state": final_state,
        }

    # ----------------------------------------------------------------
    # 生成 thinking_box 显示文本
    # ----------------------------------------------------------------

    def _build_display_info(
        self,
        intents: List[str],
        route_results: dict,
        rag_final_state: dict,
    ) -> str:
        """拼装展示在 thinking_box 中的状态文字。"""
        label_map = {
            "RAG":         "📚 知识库检索",
            "MCP_JOB":     "💼 岗位实时查询",
            "MCP_COMPANY": "🏢 公司评价查询",
            "DIRECT":      "💬 直接回答",
        }
        route_labels = " + ".join(label_map.get(i, i) for i in intents)
        display = f"**判定意图：** {route_labels}"

        # RAG 反思信息
        if rag_final_state:
            reflection_log = rag_final_state.get("reflection_log", [])
            steps = rag_final_state.get("steps_count", 0)
            score = rag_final_state.get("faithful_score", 0.0)
            status_msg = rag_final_state.get("retrieval_status", "")
            if reflection_log:
                display += f"\n*(RAG 经过 {steps} 轮反思优化，忠实度={score:.2f})*"
            if status_msg:
                display += f"\n*({status_msg})*"

        # MCP 占位提示
        mcp_stubs = [
            r for r in intents
            if r in ("MCP_JOB", "MCP_COMPANY")
            and route_results.get(r, {}).get("status") == "stub"
        ]
        if mcp_stubs:
            display += "\n⚠️ 岗位/公司实时数据当前为**占位模式**，已基于常识回答"

        # MCP 超时/错误提示
        mcp_issues = [
            r for r in intents
            if r in ("MCP_JOB", "MCP_COMPANY")
            and route_results.get(r, {}).get("status") in ("timeout", "error")
        ]
        if mcp_issues:
            display += f"\n⚠️ {'/'.join(mcp_issues)} 数据获取异常，已降级处理"

        return display

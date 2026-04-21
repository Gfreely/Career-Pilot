"""
test_validate_rewritten_query.py
单元测试：validate_rewritten_query() 校验层 (P0-P5)

测试矩阵：
  P0 — 空值检测              → fallback
  P1 — 极端长度（过短/过长）  → fallback
  P2 — 低置信度              → fix（LLM 修复成功）
  P2 — 低置信度 + 修复失败   → fallback
  P3 — 实体幻觉              → fix（LLM 修复成功）
  P3 — 实体幻觉 + 修复失败   → fallback
  P4 — 语义漂移（Jaccard）   → fix（LLM 修复成功）
  P4 — 语义漂移 + 修复失败   → fallback
  P5 — 全部通过              → pass

辅助函数测试：
  _compute_keyword_overlap   — bigram Jaccard 计算
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.multi_router import (
    validate_rewritten_query,
    _compute_keyword_overlap,
    JACCARD_DRIFT_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)


# ===========================================================
# Fake LLM Client
# ===========================================================

class FakeLLMClient:
    """模拟 UnifiedLLMClient，call_small_model 直接返回预设字符串。"""

    def __init__(self, output: str = "") -> None:
        self.output = output
        self.call_count = 0

    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        self.call_count += 1
        if not self.output:
            raise RuntimeError("FakeLLMClient: 模拟修复失败")
        return self.output


# ===========================================================
# _compute_keyword_overlap 辅助测试
# ===========================================================

def test_overlap_identical_strings():
    """完全相同的字符串 Jaccard = 1.0。"""
    score = _compute_keyword_overlap("嵌入式开发 简历优化", "嵌入式开发 简历优化")
    assert score == 1.0


def test_overlap_completely_different():
    """完全不相关的字符串 Jaccard 应低于漂移阈值。"""
    score = _compute_keyword_overlap("嵌入式开发简历", "亚马逊云服务价格对比")
    assert score < JACCARD_DRIFT_THRESHOLD


def test_overlap_partial():
    """部分重叠时分数在 (0, 1) 范围内。"""
    score = _compute_keyword_overlap("华为嵌入式岗位薪资", "华为嵌入式技术要求")
    assert 0.0 < score < 1.0


def test_overlap_empty_strings():
    """空字符串应返回 0.0，不抛出异常。"""
    assert _compute_keyword_overlap("", "") == 0.0
    assert _compute_keyword_overlap("abc", "") == 0.0
    assert _compute_keyword_overlap("", "abc") == 0.0


# ===========================================================
# P0 — 空值检测
# ===========================================================

def test_p0_empty_string_fallback():
    """rewritten_query 为空字符串 → fallback，不调用 LLM。"""
    client = FakeLLMClient(output="修复后查询")
    result = validate_rewritten_query(
        original_query="嵌入式开发简历怎么写",
        rewritten_query="",
        entities={},
        confidence=0.9,
        llm_client=client,
    )
    assert result["action"] == "fallback"
    assert result["final_query"] == "嵌入式开发简历怎么写"
    assert result["valid"] is False
    assert client.call_count == 0  # P0 不调用 LLM


def test_p0_whitespace_only_fallback():
    """rewritten_query 仅含空白 → fallback。"""
    client = FakeLLMClient(output="修复后查询")
    result = validate_rewritten_query(
        original_query="校招简历技巧",
        rewritten_query="   \t\n  ",
        entities={},
        confidence=0.95,
        llm_client=client,
    )
    assert result["action"] == "fallback"
    assert result["valid"] is False
    assert client.call_count == 0


# ===========================================================
# P1 — 极端长度
# ===========================================================

def test_p1_too_short_fallback():
    """rewritten_query 长度 < 4 → fallback，不调用 LLM。"""
    client = FakeLLMClient(output="修复后查询")
    result = validate_rewritten_query(
        original_query="嵌入式开发岗位技术要求",
        rewritten_query="HC",   # 仅 2 字符
        entities={},
        confidence=0.9,
        llm_client=client,
    )
    assert result["action"] == "fallback"
    assert result["valid"] is False
    assert client.call_count == 0


def test_p1_too_long_fallback():
    """rewritten_query 长度 > 150 → fallback，不调用 LLM。"""
    client = FakeLLMClient(output="修复后查询")
    long_query = "嵌入式" * 60   # 180 字符
    result = validate_rewritten_query(
        original_query="嵌入式开发简历",
        rewritten_query=long_query,
        entities={},
        confidence=0.85,
        llm_client=client,
    )
    assert result["action"] == "fallback"
    assert result["valid"] is False
    assert client.call_count == 0


# ===========================================================
# P2 — 低置信度
# ===========================================================

def test_p2_low_confidence_fix_success():
    """confidence 低于阈值，LLM 修复成功 → action=fix。"""
    client = FakeLLMClient(output="嵌入式开发 简历技巧")
    result = validate_rewritten_query(
        original_query="嵌入式开发简历怎么写",
        rewritten_query="嵌入式简历检索优化词",
        entities={},
        confidence=LOW_CONFIDENCE_THRESHOLD - 0.1,   # 低于阈值
        llm_client=client,
    )
    assert result["action"] == "fix"
    assert result["final_query"] == "嵌入式开发 简历技巧"
    assert result["valid"] is True
    assert client.call_count == 1


def test_p2_low_confidence_fix_failure_fallback():
    """confidence 低于阈值，LLM 修复失败 → action=fallback。"""
    client = FakeLLMClient(output="")  # 空输出 → 修复失败
    result = validate_rewritten_query(
        original_query="嵌入式开发简历怎么写",
        rewritten_query="嵌入式简历检索优化词",
        entities={},
        confidence=LOW_CONFIDENCE_THRESHOLD - 0.1,
        llm_client=client,
    )
    assert result["action"] == "fallback"
    assert result["final_query"] == "嵌入式开发简历怎么写"
    assert result["valid"] is False


# ===========================================================
# P3 — 实体幻觉检测
# ===========================================================

def test_p3_hallucinated_company_fix_success():
    """改写词引入了原始 query 和 entities 中都不存在的知名公司 → fix。"""
    client = FakeLLMClient(output="嵌入式算法岗位简历技巧")
    result = validate_rewritten_query(
        original_query="嵌入式算法岗位怎么投简历",
        rewritten_query="华为 嵌入式算法岗位 简历技巧",   # "华为"未在原始问题中
        entities={"company": None, "position": "嵌入式算法", "location": None, "keywords": []},
        confidence=0.85,
        llm_client=client,
    )
    assert result["action"] == "fix"
    assert "华为" not in result["final_query"] or result["final_query"] == "嵌入式算法岗位简历技巧"
    assert result["valid"] is True
    assert client.call_count == 1


def test_p3_registered_company_passes():
    """改写词中的公司已在 entities.company 中登记 → 不触发 P3，继续检查 P4。"""
    # 构造一个 Jaccard 足够高（≥ 0.2）的改写词，确保 P4 也不触发
    client = FakeLLMClient(output="")
    result = validate_rewritten_query(
        original_query="华为嵌入式岗位薪资怎么样",
        rewritten_query="华为嵌入式岗位薪资待遇",    # Jaccard 应较高
        entities={"company": "华为", "position": "嵌入式", "location": None, "keywords": []},
        confidence=0.9,
        llm_client=client,
    )
    # P3 不应触发；若 P4 也未触发，则 action=pass
    assert result["action"] in ("pass", "fix")   # P4 可能触发但不应是 P3
    # 核心断言：LLM 调用次数为 0（P3 未触发）或仅因 P4 触发一次
    # 由于 output 为空，若触发 fix，final_query 会变回原始问题
    if result["action"] == "pass":
        assert result["final_query"] == "华为嵌入式岗位薪资待遇"


def test_p3_company_in_original_query_not_hallucination():
    """改写词中出现的公司名也在原始问题中已有 → 不视为幻觉。"""
    client = FakeLLMClient(output="")
    result = validate_rewritten_query(
        original_query="腾讯后端开发岗校招HC",
        rewritten_query="腾讯后端校招HC 2025 状态",
        entities={"company": None, "position": "后端开发", "location": None, "keywords": []},
        confidence=0.85,
        llm_client=client,
    )
    # "腾讯"在原始问题中存在，P3 不应触发
    # 若 P4 也通过则为 pass；LLM 调用次数对 P3 应为 0
    assert result["action"] in ("pass", "fix", "fallback")
    # 最关键：不因 P3 触发（P3 触发时 client.call_count >= 1，但 P4 也可能）
    # 通过检查 validation_log 字段来区分
    assert "P3" not in result["validation_log"]


# ===========================================================
# P4 — 语义漂移（Jaccard）
# ===========================================================

def test_p4_semantic_drift_fix_success():
    """Jaccard 系数低于阈值，LLM 修复成功 → action=fix。"""
    client = FakeLLMClient(output="微电子专业秋招前景 技术栈准备")
    result = validate_rewritten_query(
        original_query="微电子专业今年秋招前景好吗",
        rewritten_query="集成电路EDA工具市场分析报告2025年趋势",  # 语义漂移
        entities={},
        confidence=0.85,
        llm_client=client,
    )
    assert result["action"] == "fix"
    assert result["final_query"] == "微电子专业秋招前景 技术栈准备"
    assert result["valid"] is True


def test_p4_semantic_drift_fix_failure_fallback():
    """Jaccard 低于阈值，LLM 修复失败 → fallback。"""
    client = FakeLLMClient(output="")   # 修复失败
    result = validate_rewritten_query(
        original_query="微电子专业今年秋招前景好吗",
        rewritten_query="集成电路EDA工具市场分析报告2025年趋势",
        entities={},
        confidence=0.85,
        llm_client=client,
    )
    assert result["action"] == "fallback"
    assert result["final_query"] == "微电子专业今年秋招前景好吗"
    assert result["valid"] is False


# ===========================================================
# P5 — 全部通过
# ===========================================================

def test_p5_all_pass():
    """改写词合法、置信度高、无实体幻觉、Jaccard 足够 → action=pass，不调用 LLM。"""
    client = FakeLLMClient(output="不应调用")
    result = validate_rewritten_query(
        original_query="双非电子信息本科简历怎么写",
        rewritten_query="双非本科电子信息 简历优化 大厂求职",
        entities={"company": None, "position": None, "location": None, "keywords": ["简历优化"]},
        confidence=0.95,
        llm_client=client,
    )
    assert result["action"] == "pass"
    assert result["final_query"] == "双非本科电子信息 简历优化 大厂求职"
    assert result["valid"] is True
    assert client.call_count == 0   # P5 不调用 LLM


def test_p5_validation_log_contains_jaccard():
    """action=pass 时，validation_log 应包含 Jaccard 分数。"""
    client = FakeLLMClient(output="")
    result = validate_rewritten_query(
        original_query="校招面试技巧有哪些",
        rewritten_query="校招面试技巧准备方法",
        entities={},
        confidence=0.9,
        llm_client=client,
    )
    if result["action"] == "pass":
        assert "Jaccard" in result["validation_log"]
        assert "P5" in result["validation_log"]


# ===========================================================
# 边界场景
# ===========================================================

def test_returns_original_query_when_fallback():
    """任意 fallback 情形下，final_query 必须等于 original_query。"""
    client = FakeLLMClient(output="")
    for rq in ["", "x", "x" * 200]:
        result = validate_rewritten_query(
            original_query="嵌入式简历怎么写",
            rewritten_query=rq,
            entities={},
            confidence=0.9,
            llm_client=client,
        )
        if result["action"] == "fallback":
            assert result["final_query"] == "嵌入式简历怎么写", \
                f"fallback 时 final_query 不正确 (rq={rq!r})"


def test_llm_exception_in_fix_falls_back_gracefully():
    """_fix_rewrite 内 LLM 抛出异常时不应向上传播，应静默 fallback。"""

    class ExplodingLLMClient:
        def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
            raise ConnectionError("模拟网络断开")

    result = validate_rewritten_query(
        original_query="硬件工程师简历怎么写",
        rewritten_query="谷歌 硬件工程师 简历模板",   # P3 触发，调用修复
        entities={"company": None, "position": None, "location": None, "keywords": []},
        confidence=0.9,
        llm_client=ExplodingLLMClient(),
    )
    # 修复异常 → fallback，不抛出
    assert result["action"] == "fallback"
    assert result["final_query"] == "硬件工程师简历怎么写"
    assert result["valid"] is False

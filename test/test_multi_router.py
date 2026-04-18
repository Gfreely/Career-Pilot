import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.multi_router import analyse_query


class FakeLLMClient:
    def __init__(self, output: str) -> None:
        self.output = output

    def call_small_model(self, system_prompt: str, user_query: str = "") -> str:
        return self.output


def test_analyse_query_repairs_missing_comma_json():
    """验证 analyse_query 可以修复字段间缺失英文逗号的常见坏 JSON。"""
    raw = """{
  "reasoning": "问候语，应该直接回答",
  "intents": ["DIRECT"]
  "rewritten_query": "",
  "entities": {
    "company": null,
    "position": null,
    "location": null,
    "keywords": []
  },
  "confidence": 0.93
}"""

    result = analyse_query("你好", FakeLLMClient(raw))

    assert result["intents"] == ["DIRECT"]
    assert result["rewritten_query"] == ""
    assert result["confidence"] == 0.93


def test_analyse_query_uses_direct_heuristic_when_json_is_unrecoverable():
    """验证 JSON 无法修复时，基础问候会回退到 DIRECT，避免误触发 RAG。"""
    result = analyse_query("你好", FakeLLMClient("这不是 JSON"))

    assert result["intents"] == ["DIRECT"]
    assert result["rewritten_query"] == ""


def test_analyse_query_salvages_schema_when_string_contains_unescaped_quotes():
    """验证字段值里出现裸引号时，仍可按固定 schema 恢复关键路由字段。"""
    raw = """{
  "reasoning": "用户在问 "华为" 嵌入式岗位的技能与待遇",
  "intents": ["RAG", "MCP_COMPANY"],
  "rewritten_query": "华为 "嵌入式" 岗位 技术要求 薪资待遇",
  "entities": {
    "company": "华为",
    "position": "嵌入式开发",
    "location": null,
    "keywords": ["嵌入式", "薪资"]
  },
  "confidence": 0.91
}"""

    result = analyse_query("华为嵌入式岗位技术要求和薪资怎么样", FakeLLMClient(raw))

    assert result["intents"] == ["RAG", "MCP_COMPANY"]
    assert result["entities"]["company"] == "华为"
    assert result["entities"]["position"] == "嵌入式开发"
    assert result["entities"]["keywords"] == ["嵌入式", "薪资"]
    assert result["confidence"] == 0.91

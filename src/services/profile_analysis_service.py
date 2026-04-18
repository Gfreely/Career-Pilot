from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.core.llm_client import UnifiedLLMClient
from src.memory import MemoryManager


PROFILE_ANALYSIS_PROMPT = """
你是一名电子信息与计算机相关方向的资深求职顾问。请基于用户画像、上传的简历内容（如果有）、最近对话上下文和目标岗位信息，输出一份结构化的求职分析报告。

要求：
1. 结论必须具体、可执行，禁止空泛表述。
2. 要同时指出优势、短板、风险和改进动作。
3. 输出必须是严格 JSON，不要输出 Markdown 代码块，不要输出额外说明。
4. match_score 取值范围 0-100。
5. suggested_roles、strengths、gaps、risks、action_plan、interview_focus 都必须是数组。

输出格式：
{{
  "summary": "总体判断",
  "match_score": 0,
  "strengths": ["优势1", "优势2"],
  "gaps": ["短板1", "短板2"],
  "risks": ["风险1", "风险2"],
  "action_plan": ["行动1", "行动2"],
  "suggested_roles": ["岗位1", "岗位2"],
  "interview_focus": ["面试准备点1", "面试准备点2"]
}}

用户画像：
{profile_text}

用户简历：
{resume_content}

最近对话：
{recent_context}

目标信息：
- 目标岗位：{target_position}
- 目标城市：{target_city}
- 目标方向：{target_direction}
- 用户补充说明：{notes}
"""


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON 对象。"""
    if not raw_text:
        raise ValueError("模型未返回分析结果")

    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("模型返回结果不是合法 JSON 对象")
    return json.loads(match.group(0))


class ProfileAnalysisService:
    """用户简历分析服务。"""

    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None) -> None:
        self.llm_client = llm_client or UnifiedLLMClient()

    def analyze(
        self,
        conversation_manager: MemoryManager,
        model_name: str,
        target_position: str,
        target_city: str = "",
        target_direction: str = "",
        notes: str = "",
        resume_content: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """基于画像和最近对话生成结构化分析报告。"""
        profile_text = conversation_manager.get_profile_text() or "暂无结构化画像信息"
        recent_context = self._build_recent_context(conversation_manager, conversation_id)

        prompt = PROFILE_ANALYSIS_PROMPT.format(
            profile_text=profile_text,
            resume_content=resume_content or "未提供简历信息",
            recent_context=recent_context,
            target_position=target_position or "未指定",
            target_city=target_city or "未指定",
            target_direction=target_direction or "未指定",
            notes=notes or "无",
        )

        response = self.llm_client.call_large_model(
            messages=[{"role": "user", "content": prompt}],
            model_name=model_name,
            stream=False
        )
        raw = response.choices[0].message.content.strip()
        result = parse_json_object(raw)
        return self._normalize_result(result)

    def _build_recent_context(
        self,
        conversation_manager: MemoryManager,
        conversation_id: Optional[str],
    ) -> str:
        """构造最近对话文本。"""
        if not conversation_id:
            conversation_id = conversation_manager.current_conversation_id
        if not conversation_id:
            return "无"

        history = conversation_manager.get_short_term_context(conversation_id, max_turns=5)
        if not history:
            return "无"

        lines = []
        for item in history:
            lines.append(f"{item['role']}: {item['content']}")
        return "\n".join(lines)

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """清洗模型输出，保证字段完整。"""
        def ensure_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(item) for item in value if str(item).strip()]
            if value:
                return [str(value)]
            return []

        score = result.get("match_score", 0)
        try:
            score = int(score)
        except Exception:
            score = 0
        score = max(0, min(100, score))

        return {
            "summary": str(result.get("summary", "")).strip(),
            "match_score": score,
            "strengths": ensure_list(result.get("strengths")),
            "gaps": ensure_list(result.get("gaps")),
            "risks": ensure_list(result.get("risks")),
            "action_plan": ensure_list(result.get("action_plan")),
            "suggested_roles": ensure_list(result.get("suggested_roles")),
            "interview_focus": ensure_list(result.get("interview_focus")),
        }

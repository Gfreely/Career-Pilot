from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.core.llm_client import UnifiedLLMClient
from src.memory import MemoryManager
from src.utils.json_utils import parse_json_object


INTERVIEW_QUESTION_PROMPT = """
你是一名电子信息领域的资深面试官，请基于用户画像、目标岗位和要求的题型，生成结构化面试题。

要求：
1. 输出必须是严格 JSON，不要输出 Markdown，不要输出多余解释。
2. questions 必须是数组，每个元素都必须包含：
   - question
   - question_type
   - focus
   - reference_answer
   - follow_up
   - reason
3. 题目要与用户画像和目标岗位相关，避免泛泛而谈。
4. 难度只允许结合用户要求输出，不要忽略。
5. question_count 必须与实际输出题目数量一致。

输出格式：
{{
  "target_position": "岗位名",
  "difficulty": "难度",
  "question_count": 0,
  "questions": [
    {{
      "question": "题目",
      "question_type": "题型",
      "focus": "考察点",
      "reference_answer": "参考答案",
      "follow_up": "追问",
      "reason": "为什么针对该用户生成这道题"
    }}
  ]
}}

用户画像：
{profile_text}

最近对话：
{recent_context}

目标信息：
- 目标岗位：{target_position}
- 难度：{difficulty}
- 题量：{question_count}
- 题型：{question_types}
- 用户补充说明：{notes}
"""





class InterviewQuestionService:
    """面试题生成服务。"""

    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None) -> None:
        self.llm_client = llm_client or UnifiedLLMClient()

    def generate_questions(
        self,
        conversation_manager: MemoryManager,
        model_name: str,
        target_position: str,
        difficulty: str = "中等",
        question_count: int = 5,
        question_types: Optional[List[str]] = None,
        notes: str = "",
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成结构化面试题集合。"""
        profile_text = conversation_manager.get_profile_text() or "暂无结构化画像信息"
        recent_context = self._build_recent_context(conversation_manager, conversation_id)
        types = question_types or ["综合问答"]

        prompt = INTERVIEW_QUESTION_PROMPT.format(
            profile_text=profile_text,
            recent_context=recent_context,
            target_position=target_position or "未指定",
            difficulty=difficulty or "中等",
            question_count=max(1, int(question_count)),
            question_types="、".join(types),
            notes=notes or "无",
        )

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.llm_client.call_large_model(
                    messages=[{"role": "user", "content": prompt}],
                    model_name=model_name,
                    stream=False
                )
                raw = response.choices[0].message.content.strip()
                result = parse_json_object(raw)
                return self._normalize_result(result, target_position, difficulty, max(1, int(question_count)))
            except Exception as e:
                last_error = e
                # 触发重试惩罚补丁：将错误明确告知 LLM，逼迫其反思纠正
                prompt += f"\n\n[系统警告]: 你上一次生成的格式无法被 JSON 解析，报错提示为 '{str(e)}'。请仔细检查你的输出！确保不要遗漏逗号，不要多重嵌套大括号，仅返回严格的整块 JSON 对象内容。"

        raise ValueError(f"连续 {max_retries} 次调用模型均未生成合法 JSON，最后一次错误: {last_error}")

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

    def _normalize_result(
        self,
        result: Dict[str, Any],
        target_position: str,
        difficulty: str,
        question_count: int,
    ) -> Dict[str, Any]:
        """清洗模型输出，保证结构稳定。"""
        normalized_questions = []
        for item in result.get("questions", []):
            if not isinstance(item, dict):
                continue
            normalized_questions.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "question_type": str(item.get("question_type", "")).strip(),
                    "focus": str(item.get("focus", "")).strip(),
                    "reference_answer": str(item.get("reference_answer", "")).strip(),
                    "follow_up": str(item.get("follow_up", "")).strip(),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )

        return {
            "target_position": str(result.get("target_position", target_position)).strip() or target_position,
            "difficulty": str(result.get("difficulty", difficulty)).strip() or difficulty,
            "question_count": len(normalized_questions) if normalized_questions else question_count,
            "questions": normalized_questions,
        }

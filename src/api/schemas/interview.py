from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class InterviewQuestionItem(BaseModel):
    question: str
    question_type: str
    focus: str
    reference_answer: str
    follow_up: str
    reason: str


class InterviewQuestionGenerateRequest(BaseModel):
    model_name: str = "Qwen/Qwen3.5-397B-A17B"
    target_position: str
    difficulty: str = "中等"
    question_count: int = 5
    question_types: List[str] = Field(default_factory=lambda: ["综合问答"])
    notes: str = ""
    conversation_id: Optional[str] = None


class InterviewQuestionGenerateResponse(BaseModel):
    target_position: str
    difficulty: str
    question_count: int
    questions: List[InterviewQuestionItem] = Field(default_factory=list)

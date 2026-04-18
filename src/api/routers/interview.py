from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_interview_question_service, get_memory_manager
from src.api.schemas.interview import (
    InterviewQuestionGenerateRequest,
    InterviewQuestionGenerateResponse,
)
from src.memory import MemoryManager
from src.services import InterviewQuestionService


router = APIRouter(prefix="/api/interview", tags=["interview"])


@router.post("/questions/generate", response_model=InterviewQuestionGenerateResponse)
def generate_interview_questions(
    payload: InterviewQuestionGenerateRequest,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    interview_question_service: InterviewQuestionService = Depends(get_interview_question_service),
):
    """基于用户画像生成结构化面试题。"""
    return interview_question_service.generate_questions(
        conversation_manager=conversation_manager,
        model_name=payload.model_name,
        target_position=payload.target_position,
        difficulty=payload.difficulty,
        question_count=payload.question_count,
        question_types=payload.question_types,
        notes=payload.notes,
        conversation_id=payload.conversation_id,
    )

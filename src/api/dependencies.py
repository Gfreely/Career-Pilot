from __future__ import annotations

from functools import lru_cache

from src.memory import MemoryManager
from src.services import (
    ChatService,
    InterviewQuestionService,
    ProfileAnalysisService,
    ProfileService,
)


def get_memory_manager() -> MemoryManager:
    """为每次请求创建独立的内存管理器，避免共享可变状态。"""
    return MemoryManager()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """返回聊天服务单例，重资源按需初始化。"""
    return ChatService()


@lru_cache(maxsize=1)
def get_profile_service() -> ProfileService:
    """返回画像服务单例。"""
    return ProfileService()


@lru_cache(maxsize=1)
def get_profile_analysis_service() -> ProfileAnalysisService:
    """返回用户建立分析服务单例。"""
    return ProfileAnalysisService()


@lru_cache(maxsize=1)
def get_interview_question_service() -> InterviewQuestionService:
    """返回面试题生成服务单例。"""
    return InterviewQuestionService()

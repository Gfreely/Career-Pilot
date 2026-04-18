from __future__ import annotations

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_chat_service, get_memory_manager
from src.api.schemas.chat import (
    ChatReplyRequest,
    ChatReplyResponse,
    ConversationDetailResponse,
    ConversationListItem,
    CreateConversationRequest,
)
from src.memory import MemoryManager
from src.services import ChatService


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/conversations", response_model=list[ConversationListItem])
def list_conversations(conversation_manager: MemoryManager = Depends(get_memory_manager)):
    """列出全部对话。"""
    return conversation_manager.get_all_conversations()


@router.post("/conversations", response_model=ConversationDetailResponse)
def create_conversation(
    payload: CreateConversationRequest,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
):
    """创建新对话。"""
    conversation_id = conversation_manager.create_conversation(payload.title)
    return conversation_manager.get_conversation(conversation_id)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: str,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
):
    """获取指定对话详情。"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


@router.post("/conversations/{conversation_id}/reply", response_model=ChatReplyResponse)
def reply_conversation(
    conversation_id: str,
    payload: ChatReplyRequest,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    chat_service: ChatService = Depends(get_chat_service),
):
    """向指定对话发送消息并获取同步回复。"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    conversation_manager.current_conversation_id = conversation_id
    thinking = ""
    content = ""

    for thinking, content in chat_service.generate_response(
        message=payload.message,
        conversation_manager=conversation_manager,
        stream_mode=False,
        model=payload.model,
        prompt_template=payload.prompt_template,
    ):
        pass

    return {
        "conversation_id": conversation_id,
        "thinking": thinking,
        "content": content,
    }


@router.post("/conversations/{conversation_id}/stream")
def stream_conversation(
    conversation_id: str,
    payload: ChatReplyRequest,
    conversation_manager: MemoryManager = Depends(get_memory_manager),
    chat_service: ChatService = Depends(get_chat_service),
):
    """向指定对话发送消息并获取流式回复(SSE)。"""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")

    conversation_manager.current_conversation_id = conversation_id

    def event_stream():
        for thinking, content in chat_service.generate_response(
            message=payload.message,
            conversation_manager=conversation_manager,
            stream_mode=True,
            model=payload.model,
            prompt_template=payload.prompt_template,
        ):
            # SSE 要求必须严格按 `data: {}\n\n` 格式
            chunk = {
                "thinking": thinking,
                "content": content,
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

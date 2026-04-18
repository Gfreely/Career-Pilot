from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    working_memory: Dict[str, Any] = Field(default_factory=dict)


class ChatReplyRequest(BaseModel):
    message: str
    model: str = "Pro/MiniMaxAI/MiniMax-M2.5"
    prompt_template: str = "default"


class ChatReplyResponse(BaseModel):
    conversation_id: str
    thinking: str
    content: str

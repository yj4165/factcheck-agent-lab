from __future__ import annotations

import uuid
from typing import List, Literal, Union

from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    """대화 안의 텍스트 메시지"""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str

class ToolCall(BaseModel):
    """도구를 실행해 달라는 LLM의 요청"""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict

class ToolResult(BaseModel):
    """도구 실행 결과"""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list

ContentItem = Union[Message, ToolCall, ToolResult]

class Event(BaseModel):
    """에이전트 실행 중 기록된 사건"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str # "user" 또는 에이전트 이름
    content: List[ContentItem] = Field(default_factory=list)
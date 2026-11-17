from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from litellm import acompletion
from pydantic import BaseModel, Field

from agent.tools.base import BaseTool
from agent.types import ContentItem, Message, ToolCall, ToolResult

class LlmRequest(BaseModel):
    """LLM 호출을 위한 요청 객체"""
    model_config = {"arbitrary_types_allowed": True}
    
    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    # tool_choice: Optional[str] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class LlmResponse(BaseModel):
    """LLM 호출에서 돌아온 응답 객체"""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)

class LlmClient:
    """LiteLLM을 사용해 LLM API를 호출하는 클라이언트"""

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """LLM에서 응답을 생성한다."""
        try:
            messages = self._build_messages(request)
            tools = [t.tool_definition for t in request.tools] if request.tools else None

            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice}
                   if request.tool_choice else {}),
                **self.config
            )

            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))

    def _build_messages(self, request: LlmRequest) -> List[dict]:
        """LlmRequest를 API 메시지 형식으로 변환한다."""
        messages = []

        for instruction in request.instructions:
            messages.append({"role": "system", "content": instruction})

        for item in request.contents:
            if isinstance(item, Message):
                messages.append({"role": item.role, "content": item.content})
            elif isinstance(item, ToolCall):
                tool_call_dict = {
                    "id": item.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": json.dumps(item.arguments)
                    }
                }
                # 이전 assistant 메시지가 있으면 거기에 추가한다.
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(tool_call_dict)
                else:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call_dict]
                    })

            elif isinstance(item, ToolResult):
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": str(item.content[0]) if item.content else ""
                })

        return messages

    def _parse_response(self, response) -> LlmResponse:
        """API 응답을 LlmResponse로 변환한다."""
        choice = response.choices[0]
        content_items = []

        if choice.message.content:
            content_items.append(Message(
                role="assistant",
                content=choice.message.content
            ))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_items.append(ToolCall(
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LlmResponse(
            content=content_items,
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        )
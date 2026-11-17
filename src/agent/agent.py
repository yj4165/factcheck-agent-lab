
from typing import List, Optional, Type

from pydantic import BaseModel

from agent.llm import LlmClient, LlmRequest, LlmResponse
from agent.tools.base import BaseTool, FunctionTool
from agent.context import ExecutionContext, AgentResult
from agent.types import Event, Message, ToolCall, ToolResult
from agent.tools.helpers import format_tool_definition


class Agent:
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        output_type: Optional[Type[BaseModel]] = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.output_type = output_type
        self.output_tool_name: str | None = None
        self.tools = self._setup_tools(tools or [])


    async def run(
        self,
        user_input: str,
        context: ExecutionContext = None,
    ) -> AgentResult:
        # 컨텍스트를 만들거나 재사용한다.
        if context is None:
            context = ExecutionContext()

        # 사용자 입력을 첫 번째 이벤트로 추가한다.
        user_event = Event(
            execution_id=context.execution_id,
            author="user",
            content=[Message(role="user", content=user_input)]
        )
        context.add_event(user_event)

        # 완료되거나 최대 단계 수에 도달할 때까지 단계를 실행한다.
        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context)

            # 마지막 이벤트가 최종 응답인지 확인한다.
            last_event = context.events[-1]
            if self._is_final_response(last_event):
                context.final_result = self._extract_final_result(last_event)

        return AgentResult(output=context.final_result, context=context)


    async def step(self, context: ExecutionContext):
        # LLM에 보낼 내용을 준비한다.
        llm_request = self._prepare_llm_request(context)

        # LLM의 결정을 받는다.
        llm_response = await self.think(llm_request)

        # LLM 응답을 이벤트로 기록한다.
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        # LLM이 요청한 도구가 있으면 실행한다.
        tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            tool_results = await self.act(context, tool_calls)
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=tool_results
            )
            context.add_event(tool_event)

        context.increment_step()

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: List[ToolCall],
    ) -> List[ToolResult]:
        tools_dict = {tool.name: tool for tool in self.tools}
        results = []

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[f"Tool '{tool_call.name}' not found"],
                ))
                continue

            tool = tools_dict[tool_call.name]

            try:
                output = await tool(context, **tool_call.arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output]
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)]
                ))

        return results

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        tools = list(tools)

        if self.output_type is not None:
            output_schema = self.output_type.model_json_schema()
            output_schema.pop("title", None)
            # output_schema.pop("$defs", None)
            
            tool_definition = format_tool_definition(
                "final_answer",
                "Return the final structured answer matching the required schema.",
                output_schema,
            )

            captured_type = self.output_type

            def _parse_output(**kwargs) -> BaseModel:
                return captured_type.model_validate(kwargs)
                

            final_answer_tool = FunctionTool(
                func=_parse_output,
                name="final_answer",
                description="Return the final structured answer matching the required schema.",
                tool_definition=tool_definition,
            )
            tools.append(final_answer_tool)
            self.output_tool_name = "final_answer"
            
        return tools
    
    def _prepare_llm_request(self, context:ExecutionContext) -> LlmRequest:
        # 이벤트를 콘텐츠 항목으로 펼친다.
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        if self.output_tool_name and context.current_step >= self.max_steps -1:
            tool_choice = {"type": "function", "function": {"name": self.output_tool_name}}
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    def _is_final_response(self, event: Event) -> bool:
        if self.output_tool_name:
            # 구조화된 출력에서는 final_answer 도구가 성공했는지 확인한다.
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                ):
                    return True
            return False
            
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> str | BaseModel:
        if self.output_tool_name:
            # final_answer 도구 결과에서 구조화된 출력을 추출한다.
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and item.content
                ):
                    return item.content[0]

        # 자유 텍스트 응답을 위한 기존 로직
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

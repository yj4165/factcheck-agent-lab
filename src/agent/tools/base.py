from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from agent.context import ExecutionContext
from agent.tools.helpers import function_to_input_schema, format_tool_definition

class BaseTool(ABC):
    """모든 도구를 위한 추상 기본 클래스"""

    def __init__(
        self,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """파이썬 함수를 BaseTool로 감싼다."""

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None
    ):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters

        super().__init__(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip(),
            tool_definition=tool_definition,
        )
        
        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """감싼 함수를 실행한다."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        # 동기 함수와 비동기 함수를 모두 처리한다.
        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        """함수 시그니처에서 도구 정의를 생성한다."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)

def tool(func=None, *, name=None, description=None):

    def decorator(f):
        return FunctionTool(
            func=f,
            name=name,
            description=description,
        )

    if func is not None:
        return decorator(func)
    return decorator
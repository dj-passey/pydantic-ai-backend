"""Console capability for pydantic-ai agents.

Provides ``ConsoleCapability`` that bundles console toolset + instructions +
permission enforcement via the pydantic-ai capabilities API.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_backends import ConsoleCapability
    from pydantic_ai_backends.permissions import READONLY_RULESET

    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[ConsoleCapability(permissions=READONLY_RULESET)],
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_backends.permissions.checker import PermissionChecker
from pydantic_ai_backends.permissions.types import (
    PermissionOperation,
    PermissionRuleset,
)
from pydantic_ai_backends.toolsets.console import (
    EditFormat,
    create_console_toolset,
    get_console_system_prompt,
)

# Maps tool names to permission operations
_TOOL_OPERATION_MAP: dict[str, PermissionOperation] = {
    "ls": "ls",
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "hashline_edit": "edit",
    "glob": "glob",
    "grep": "grep",
    "execute": "execute",
}

# Operations that use a file path argument for per-path permission checks
_PATH_OPERATIONS: set[PermissionOperation] = {"read", "write", "edit"}

# Operations that use a command argument for per-command permission checks
_COMMAND_OPERATIONS: set[PermissionOperation] = {"execute"}

# Note: glob, grep, ls are not in _PATH_OPERATIONS or _COMMAND_OPERATIONS.
# They don't require per-path checks — if denied, prepare_tools() hides them
# entirely. This avoids checking every glob pattern against the ruleset.


@dataclass
class ConsoleCapability(AbstractCapability[Any]):
    """Capability providing filesystem tools with permission enforcement.

    Bundles the console toolset (ls, read_file, write_file, edit_file, glob,
    grep, execute) with dynamic instructions and per-tool permission control.

    When a permission ruleset is provided:
    - Tools for denied operations are hidden from the model entirely
    - Per-path/command permissions are checked before each tool execution
    - "ask" permissions can trigger an approval callback

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_backends import ConsoleCapability
        from pydantic_ai_backends.permissions import READONLY_RULESET

        # Read-only agent — write/edit/execute tools are hidden
        agent = Agent(
            "openai:gpt-4.1",
            capabilities=[ConsoleCapability(permissions=READONLY_RULESET)],
        )
        ```
    """

    include_execute: bool = True
    """Whether to include the execute tool."""

    edit_format: EditFormat = "str_replace"
    """Edit format: 'str_replace' or 'hashline'."""

    permissions: PermissionRuleset | None = None
    """Permission ruleset for controlling tool access."""

    _toolset: AbstractToolset[Any] | None = field(default=None, init=False, repr=False)
    _checker: PermissionChecker | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the underlying console toolset and permission checker."""
        self._toolset = create_console_toolset(
            include_execute=self.include_execute,
        )
        if self.permissions is not None:
            self._checker = PermissionChecker(ruleset=self.permissions)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return name for AgentSpec YAML/JSON serialization."""
        return "ConsoleCapability"

    def get_toolset(self) -> AbstractToolset[Any] | None:
        """Return the console toolset."""
        return self._toolset

    def get_instructions(self) -> str:
        """Return console tool usage instructions."""
        return get_console_system_prompt(edit_format=self.edit_format)

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide tools for denied operations."""
        if self._checker is None:
            return tool_defs

        result = []
        for td in tool_defs:
            operation = _TOOL_OPERATION_MAP.get(td.name)
            if operation is None:
                # Not a console tool — pass through
                result.append(td)
                continue

            action = self._checker.check_sync(operation, "*")
            if action != "deny":
                result.append(td)

        return result

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Check per-path permissions before tool execution."""
        if self._checker is None:
            return args

        operation = _TOOL_OPERATION_MAP.get(call.tool_name)
        if operation is None:
            return args

        # Determine target to check
        if operation in _PATH_OPERATIONS:
            target = args.get("path", args.get("file_path", "*"))
        elif operation in _COMMAND_OPERATIONS:
            target = args.get("command", "*")
        else:
            return args

        # Check permission — raises PermissionDeniedError on deny
        await self._checker.check(operation, str(target))
        return args


__all__ = [
    "ConsoleCapability",
]

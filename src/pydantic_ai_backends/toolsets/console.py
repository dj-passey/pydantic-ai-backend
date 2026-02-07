"""Console toolset for AI agents - file operations and shell execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

from pydantic_ai import RunContext

from pydantic_ai_backends.protocol import BackendProtocol
from pydantic_ai_backends.types import GrepMatch

if TYPE_CHECKING:
    from pydantic_ai.toolsets import FunctionToolset

    from pydantic_ai_backends.permissions.types import PermissionRuleset


CONSOLE_SYSTEM_PROMPT = """
## Console Tools

You have access to console tools for file operations and command execution:

### File Operations
- `ls`: List files in a directory
- `read_file`: Read file content with line numbers
- `write_file`: Create or overwrite a file
- `edit_file`: Replace strings in a file
- `glob`: Find files matching a pattern
- `grep`: Search for patterns in files

### Shell Execution
- `execute`: Run shell commands (if enabled)

### Best Practices
- Always read a file before editing it
- Use edit_file for small changes, write_file for complete rewrites
- Use glob to find files before operating on them
- Be careful with destructive shell commands
"""


@runtime_checkable
class ConsoleDeps(Protocol):
    """Protocol for dependencies that provide a backend."""

    @property
    def backend(self) -> BackendProtocol:
        """The backend for file operations."""
        ...


class _ConsoleToolsetTestAttrs(Protocol):
    """Protocol for test-only attributes on console toolset."""

    _console_default_ignore_hidden: bool
    _console_grep_impl: Callable[..., Awaitable[str]]


def _requires_approval_from_ruleset(
    ruleset: PermissionRuleset | None,
    operation: str,
    legacy_flag: bool,
) -> bool:
    """Determine if a tool requires approval based on ruleset or legacy flag.

    If a ruleset is provided, checks if the operation's default action is "ask".
    Otherwise, falls back to the legacy boolean flag.
    """
    from pydantic_ai_backends.permissions.types import OperationPermissions

    if ruleset is None:
        return legacy_flag

    op_perms: OperationPermissions | None = getattr(ruleset, operation, None)
    if op_perms is None:
        # Use global default
        return ruleset.default == "ask"
    return op_perms.default == "ask"


def create_console_toolset(  # noqa: C901
    id: str | None = None,
    include_execute: bool = True,
    require_write_approval: bool = False,
    require_execute_approval: bool = True,
    default_ignore_hidden: bool = True,
    permissions: PermissionRuleset | None = None,
    max_retries: int = 1,
) -> FunctionToolset[ConsoleDeps]:
    """Create a console toolset for file operations and shell execution.

    This toolset provides tools for interacting with the filesystem and
    executing shell commands. It works with any backend that implements
    BackendProtocol (LocalBackend, DockerSandbox, StateBackend, etc.)

    Args:
        id: Optional unique ID for the toolset.
        include_execute: Whether to include the execute tool.
            Requires backend to have execute() method.
        require_write_approval: Whether write_file and edit_file require approval.
            Ignored if permissions is provided.
        require_execute_approval: Whether execute requires approval.
            Ignored if permissions is provided.
        default_ignore_hidden: Default behavior for grep regarding hidden files.
        permissions: Optional permission ruleset to determine tool approval requirements.
            If provided, overrides require_write_approval and require_execute_approval
            based on whether the operation's default action is "ask".
        max_retries: Maximum number of retries for each tool during a run.
            When the model sends invalid arguments (e.g. missing required fields),
            the validation error is fed back and the model can retry up to this
            many times. Defaults to 1.

    Returns:
        FunctionToolset with console tools.

    Example:
        ```python
        from dataclasses import dataclass
        from pydantic_ai_backends import LocalBackend, create_console_toolset

        @dataclass
        class MyDeps:
            backend: LocalBackend

        toolset = create_console_toolset()
        deps = MyDeps(backend=LocalBackend("/workspace"))

        # Or with permissions
        from pydantic_ai_backends.permissions import DEFAULT_RULESET

        toolset = create_console_toolset(permissions=DEFAULT_RULESET)
        ```
    """
    from pydantic_ai.toolsets import FunctionToolset

    # Determine approval requirements
    write_approval = _requires_approval_from_ruleset(permissions, "write", require_write_approval)
    execute_approval = _requires_approval_from_ruleset(
        permissions, "execute", require_execute_approval
    )

    toolset: FunctionToolset[ConsoleDeps] = FunctionToolset(id=id, max_retries=max_retries)

    @toolset.tool
    async def ls(  # pragma: no cover
        ctx: RunContext[ConsoleDeps],
        path: str = ".",
    ) -> str:
        """List files and directories at the given path.

        Args:
            path: Directory path to list. Defaults to current directory.
        """
        entries = ctx.deps.backend.ls_info(path)

        if not entries:
            return f"Directory '{path}' is empty or does not exist"

        lines = [f"Contents of {path}:"]
        for entry in entries:
            if entry["is_dir"]:
                lines.append(f"  {entry['name']}/")
            else:
                size = entry.get("size")
                size_str = f" ({size} bytes)" if size is not None else ""
                lines.append(f"  {entry['name']}{size_str}")

        return "\n".join(lines)

    @toolset.tool
    async def read_file(  # pragma: no cover
        ctx: RunContext[ConsoleDeps],
        path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            path: Path to the file to read.
            offset: Line number to start reading from (0-indexed).
            limit: Maximum number of lines to read.
        """
        return ctx.deps.backend.read(path, offset, limit)

    @toolset.tool(requires_approval=write_approval)
    async def write_file(  # pragma: no cover
        ctx: RunContext[ConsoleDeps],
        path: str,
        content: str,
    ) -> str:
        """Write content to a file (creates or overwrites).

        This will create parent directories if needed.
        Use edit_file for making small changes to existing files.

        Args:
            path: Path to the file to write.
            content: Content to write to the file.
        """
        result = ctx.deps.backend.write(path, content)

        if result.error:
            return f"Error: {result.error}"

        lines = content.count("\n") + 1
        return f"Wrote {lines} lines to {result.path}"

    @toolset.tool(requires_approval=write_approval)
    async def edit_file(  # pragma: no cover
        ctx: RunContext[ConsoleDeps],
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing strings.

        The old_string must be unique in the file unless replace_all is True.
        Always read the file first to understand its content before editing.

        Args:
            path: Path to the file to edit.
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences. Otherwise, fails if not unique.
        """
        result = ctx.deps.backend.edit(path, old_string, new_string, replace_all)

        if result.error:
            return f"Error: {result.error}"

        return f"Edited {result.path}: replaced {result.occurrences} occurrence(s)"

    @toolset.tool
    async def glob(  # pragma: no cover
        ctx: RunContext[ConsoleDeps],
        pattern: str,
        path: str = ".",
    ) -> str:
        """Find files matching a glob pattern.

        Common patterns:
        - "*.py" - Python files in current directory
        - "**/*.py" - Python files recursively
        - "src/**/*.ts" - TypeScript files under src/

        Args:
            pattern: Glob pattern to match (e.g., "**/*.py").
            path: Base directory to search from.
        """
        entries = ctx.deps.backend.glob_info(pattern, path)

        if not entries:
            return f"No files matching '{pattern}' in {path}"

        lines = [f"Found {len(entries)} file(s) matching '{pattern}':"]
        for entry in entries[:100]:
            lines.append(f"  {entry['path']}")

        if len(entries) > 100:
            lines.append(f"  ... and {len(entries) - 100} more")

        return "\n".join(lines)

    @toolset.tool
    async def grep(  # pragma: no cover
        ctx: RunContext[ConsoleDeps],
        pattern: str,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches",
        ignore_hidden: bool = default_ignore_hidden,
    ) -> str:
        """Search for a regex pattern in files.

        Args:
            pattern: Regex pattern to search for.
            path: Specific file or directory to search.
            glob_pattern: Glob pattern to filter files (e.g., "*.py").
            output_mode: Output format - "content", "files_with_matches", or "count".
            ignore_hidden: Whether to skip hidden files (defaults to toolset setting).
        """
        result = ctx.deps.backend.grep_raw(pattern, path, glob_pattern, ignore_hidden)

        if isinstance(result, str):
            return result  # Error message

        if not result:
            return f"No matches for '{pattern}'"

        matches: list[GrepMatch] = result

        if output_mode == "count":
            return f"Found {len(matches)} match(es) for '{pattern}'"

        if output_mode == "files_with_matches":
            files = sorted(set(m["path"] for m in matches))
            lines = [f"Files containing '{pattern}':"]
            for f in files[:50]:
                lines.append(f"  {f}")
            if len(files) > 50:
                lines.append(f"  ... and {len(files) - 50} more files")
            return "\n".join(lines)

        # content mode
        lines = [f"Matches for '{pattern}':"]
        for m in matches[:50]:
            lines.append(f"  {m['path']}:{m['line_number']}: {m['line'][:100]}")
        if len(matches) > 50:
            lines.append(f"  ... and {len(matches) - 50} more matches")
        return "\n".join(lines)

    # Expose references for testing
    cast(_ConsoleToolsetTestAttrs, toolset)._console_default_ignore_hidden = default_ignore_hidden
    cast(_ConsoleToolsetTestAttrs, toolset)._console_grep_impl = grep

    if include_execute:

        @toolset.tool(requires_approval=execute_approval)
        async def execute(  # pragma: no cover
            ctx: RunContext[ConsoleDeps],
            command: str,
            timeout: int | None = 120,
        ) -> str:
            """Execute a shell command.

            Use this for running tests, builds, scripts, etc.
            Be careful with destructive commands.

            Args:
                command: The shell command to execute.
                timeout: Maximum execution time in seconds (default 120).
            """
            backend = ctx.deps.backend

            # Check if backend supports execute
            if not hasattr(backend, "execute"):
                return "Error: Backend does not support command execution"

            # Check if execute is enabled (for LocalBackend)
            if hasattr(backend, "execute_enabled") and not backend.execute_enabled:  # pyright: ignore[reportAttributeAccessIssue]
                return "Error: Shell execution is disabled for this backend"

            try:
                result = backend.execute(command, timeout)  # pyright: ignore[reportAttributeAccessIssue]
            except RuntimeError as e:
                return f"Error: {e}"

            output = result.output
            if result.truncated:
                output += "\n\n... (output truncated)"

            if result.exit_code is not None and result.exit_code != 0:
                return f"Command failed (exit code {result.exit_code}):\n{output}"

            return str(output)

    return toolset


def get_console_system_prompt() -> str:
    """Get the system prompt for console tools.

    Returns:
        System prompt describing available console tools.
    """
    return CONSOLE_SYSTEM_PROMPT


# Convenience alias
ConsoleToolset = create_console_toolset

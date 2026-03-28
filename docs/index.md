# File Storage & Sandbox Backends for Pydantic AI

Console Toolset, Docker Sandbox, and Permission System for [Pydantic AI](https://ai.pydantic.dev/) agents.

---

**pydantic-ai-backend** provides file storage, sandbox execution, and a ready-to-use console toolset for [pydantic-ai](https://ai.pydantic.dev/) agents. Give your AI agents the ability to read, write, and execute code safely.

<div class="grid cards" markdown>

- :material-console: **Console Toolset**

    Ready-to-use tools: ls, read, write, edit, glob, grep, execute

- :material-docker: **Docker Isolation**

    Execute code safely in isolated containers

- :material-folder-multiple: **Multiple Backends**

    In-memory, filesystem, Docker — same interface

- :material-shield-lock: **Permission System**

    Fine-grained access control with presets

</div>

## Quick Start (Capability API)

The recommended way to add filesystem tools:

```python
from pydantic_ai import Agent
from pydantic_ai_backends import ConsoleCapability

agent = Agent("openai:gpt-4.1", capabilities=[ConsoleCapability()])
```

### With Permissions

```python
from pydantic_ai_backends import ConsoleCapability
from pydantic_ai_backends.permissions import READONLY_RULESET

# Read-only agent — write/edit/execute tools hidden from model
agent = Agent("openai:gpt-4.1", capabilities=[
    ConsoleCapability(permissions=READONLY_RULESET),
])
```

### Alternative: Toolset API

```python
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai_backends import LocalBackend, create_console_toolset

@dataclass
class Deps:
    backend: LocalBackend

agent = Agent("openai:gpt-4.1", deps_type=Deps, toolsets=[create_console_toolset()])
```

## Choose Your Backend

Same toolset, different backends — swap based on your use case:

=== "Local Development"

    ```python
    from pydantic_ai_backends import LocalBackend

    backend = LocalBackend(root_dir="./workspace")
    ```

=== "Testing"

    ```python
    from pydantic_ai_backends import StateBackend

    backend = StateBackend()  # In-memory, no side effects
    ```

=== "Production (Docker)"

    ```python
    from pydantic_ai_backends import DockerSandbox

    backend = DockerSandbox(runtime="python-datascience")
    ```

=== "Multi-User"

    ```python
    from pydantic_ai_backends import SessionManager

    manager = SessionManager(workspace_root="/app/workspaces")
    backend = await manager.get_or_create(user_id="alice")
    ```

## Available Tools

| Tool | Description |
|------|-------------|
| `ls` | List files in a directory |
| `read_file` | Read file content with line numbers |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace strings in a file |
| `glob` | Find files matching a pattern |
| `grep` | Search for patterns in files |
| `execute` | Run shell commands (optional) |

## Backend Comparison

| Backend | Persistence | Execution | Best For |
|---------|-------------|-----------|----------|
| `LocalBackend` | Persistent | Yes | CLI tools, local dev |
| `StateBackend` | Ephemeral | No | Testing, mocking |
| `DockerSandbox` | Ephemeral* | Yes | Safe execution, multi-user |
| `CompositeBackend` | Mixed | Depends | Route by path prefix |

*DockerSandbox supports persistent volumes via `workspace_root` parameter.

## Related Projects

| Package | Description |
|---------|-------------|
| [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) | Full agent framework (uses this library) |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning toolset |
| [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) | Multi-agent orchestration |
| [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) | Context management |
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | The foundation — agent framework by Pydantic |

## Next Steps

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**

    Get started with pip or uv

- :material-book-open-variant: **[Concepts](concepts/index.md)**

    Learn about backends and toolsets

- :material-code-tags: **[Examples](examples/index.md)**

    See real-world usage patterns

- :material-api: **[API Reference](api/index.md)**

    Full API documentation

</div>

# ConsoleCapability

`ConsoleCapability` is the recommended way to add filesystem tools to a Pydantic AI agent.
It's a [pydantic-ai capability](https://ai.pydantic.dev/capabilities/) that bundles console
tools, instructions, and permission enforcement.

## Why Capability over Toolset?

| Feature | ConsoleCapability | create_console_toolset |
|---------|:-:|:-:|
| Tools registered automatically | Yes | Yes |
| System prompt injected | Yes | Manual |
| Permission enforcement (deny) | Yes (`prepare_tools`) | Only `requires_approval` |
| Per-path permission checks | Yes (`before_tool_execute`) | No |
| Fixes issue #23 (READONLY) | Yes | No |

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_backends import ConsoleCapability

agent = Agent("openai:gpt-4.1", capabilities=[ConsoleCapability()])
```

## With Permissions

```python
from pydantic_ai_backends import ConsoleCapability
from pydantic_ai_backends.permissions import READONLY_RULESET, PERMISSIVE_RULESET

# Read-only — write/edit/execute tools hidden from model entirely
agent = Agent("openai:gpt-4.1", capabilities=[
    ConsoleCapability(permissions=READONLY_RULESET),
])

# Permissive — everything allowed except secrets
agent = Agent("openai:gpt-4.1", capabilities=[
    ConsoleCapability(permissions=PERMISSIVE_RULESET),
])
```

## How Permissions Work

1. **`prepare_tools`** — hides tools for denied operations. With `READONLY_RULESET`,
   the model never sees `write_file`, `edit_file`, or `execute`.

2. **`before_tool_execute`** — checks per-path permissions before each tool call.
   If a specific path is denied (e.g., `.env` files), the call is blocked even if
   the operation is generally allowed.

"""Read-only agent — write/edit/execute tools are hidden from the model."""

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_backends import ConsoleCapability, LocalBackend
from pydantic_ai_backends.permissions import READONLY_RULESET


@dataclass
class Deps:
    backend: LocalBackend


async def main() -> None:
    # With READONLY_RULESET, the model cannot see write_file, edit_file, or execute.
    # It can only read, glob, grep, and ls.
    agent = Agent(
        "openai:gpt-4.1",
        deps_type=Deps,
        capabilities=[ConsoleCapability(permissions=READONLY_RULESET)],
    )

    backend = LocalBackend(root_dir="/tmp/demo")
    result = await agent.run(
        "List all Python files and show me the contents of the first one.",
        deps=Deps(backend=backend),
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())

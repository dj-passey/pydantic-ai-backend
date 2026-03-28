"""Custom permission ruleset — fine-grained path control."""

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_backends import ConsoleCapability, LocalBackend
from pydantic_ai_backends.permissions import create_ruleset


@dataclass
class Deps:
    backend: LocalBackend


async def main() -> None:
    # Allow reads and execution, deny writes
    ruleset = create_ruleset(
        allow_read=True,
        allow_write=False,
        allow_edit=False,
        allow_execute=True,
    )

    agent = Agent(
        "openai:gpt-4.1",
        deps_type=Deps,
        capabilities=[ConsoleCapability(permissions=ruleset)],
    )

    backend = LocalBackend(root_dir="/tmp/demo")
    result = await agent.run(
        "Run pytest and tell me which tests pass.",
        deps=Deps(backend=backend),
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())

"""Basic ConsoleCapability usage — filesystem tools with one line."""

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_backends import ConsoleCapability, LocalBackend


@dataclass
class Deps:
    backend: LocalBackend


async def main() -> None:
    agent = Agent(
        "openai:gpt-4.1",
        deps_type=Deps,
        capabilities=[ConsoleCapability()],
    )

    backend = LocalBackend(root_dir="/tmp/demo")
    result = await agent.run(
        "Create a hello.py file that prints 'Hello World', then run it.",
        deps=Deps(backend=backend),
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())

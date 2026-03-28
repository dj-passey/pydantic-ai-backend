"""Multi-agent setup — different permissions per agent (issue #23 scenario)."""

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_backends import ConsoleCapability, LocalBackend
from pydantic_ai_backends.permissions import PERMISSIVE_RULESET, READONLY_RULESET


@dataclass
class Deps:
    backend: LocalBackend


async def main() -> None:
    backend = LocalBackend(root_dir="/tmp/demo")
    deps = Deps(backend=backend)

    # Agent 1: full access (can read + write + execute)
    writer = Agent(
        "openai:gpt-4.1",
        deps_type=Deps,
        capabilities=[ConsoleCapability(permissions=PERMISSIVE_RULESET)],
    )

    # Agent 2: read-only (write/edit/execute tools are hidden)
    reader = Agent(
        "openai:gpt-4.1",
        deps_type=Deps,
        capabilities=[ConsoleCapability(permissions=READONLY_RULESET)],
    )

    # Writer creates a file
    await writer.run("Write 'hello world' to /tmp/demo/test.txt", deps=deps)

    # Reader can only read — if it tries to write, the tool isn't available
    result = await reader.run("Read the contents of test.txt", deps=deps)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())

# Docker Sandbox

`DockerSandbox` provides isolated code execution for your pydantic-ai agents. Run untrusted code safely in Docker containers.

!!! warning "Requires Docker"
    ```bash
    pip install pydantic-ai-backend[docker]
    ```
    Ensure Docker is installed and the daemon is running.

## Basic Usage with pydantic-ai

```python
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai_backends import DockerSandbox, create_console_toolset

@dataclass
class Deps:
    backend: DockerSandbox

# Create sandbox with pre-configured runtime
sandbox = DockerSandbox(runtime="python-datascience")

try:
    # Add console tools to your agent
    toolset = create_console_toolset()
    agent = Agent("openai:gpt-4o", deps_type=Deps)
    agent = agent.with_toolset(toolset)

    # Agent can safely execute arbitrary code in Docker
    result = agent.run_sync(
        "Load the iris dataset with sklearn, analyze it with pandas, "
        "and create a visualization with matplotlib",
        deps=Deps(backend=sandbox),
    )
    print(result.output)
finally:
    sandbox.stop()  # Clean up container
```

## Runtime Configurations

Pre-configured environments with packages pre-installed:

```python
from pydantic_ai_backends import DockerSandbox, RuntimeConfig

# Use built-in runtime
sandbox = DockerSandbox(runtime="python-datascience")

# Or define custom runtime for your use case
runtime = RuntimeConfig(
    name="ml-env",
    base_image="python:3.12-slim",
    packages=["torch", "transformers", "pandas"],
)
sandbox = DockerSandbox(runtime=runtime)
```

### Built-in Runtimes

| Runtime | Description | Use Case |
|---------|-------------|----------|
| `python-minimal` | Clean Python 3.12 | General scripting |
| `python-datascience` | pandas, numpy, matplotlib, scikit-learn, seaborn | Data analysis |
| `python-web` | FastAPI, SQLAlchemy, httpx | Web development |
| `node-minimal` | Clean Node.js 20 | JavaScript/TypeScript |
| `node-react` | TypeScript, Vite, React | Frontend development |

## SessionManager for Multi-User

For web apps where each user needs isolated execution:

```python
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai_backends import SessionManager, DockerSandbox, create_console_toolset

@dataclass
class UserDeps:
    backend: DockerSandbox
    user_id: str

# Create session manager
manager = SessionManager(
    default_runtime="python-datascience",
    workspace_root="/app/workspaces",  # Persistent storage per user
)

async def handle_user_request(user_id: str, message: str):
    # Get or create sandbox for this user
    sandbox = await manager.get_or_create(user_id)

    # Create agent with user's isolated sandbox
    toolset = create_console_toolset()
    agent = Agent("openai:gpt-4o", deps_type=UserDeps)
    agent = agent.with_toolset(toolset)

    result = await agent.run(
        message,
        deps=UserDeps(backend=sandbox, user_id=user_id),
    )
    return result.output

# Each user's code runs in isolated container
# User A cannot see User B's files
```

### Architecture

```
                    ┌─────────────────┐
                    │ SessionManager  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│ DockerSandbox │   │ DockerSandbox │   │ DockerSandbox │
│   (User A)    │   │   (User B)    │   │   (User C)    │
│  pydantic-ai  │   │  pydantic-ai  │   │  pydantic-ai  │
│    Agent      │   │    Agent      │   │    Agent      │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Persistent Storage

By default, files are lost when container stops. Use volumes for persistence:

```python
sandbox = DockerSandbox(
    runtime="python-datascience",
    volumes={"/host/data": "/workspace/data"},  # Mount host directory
)
```

### Named Containers (Reusable)

Use `container_name` to create containers that persist between sessions.
Installed packages, caches, and filesystem state survive restarts:

```python
sandbox = DockerSandbox(
    image="python:3.12-slim",
    container_name="my-dev-env",  # implies auto_remove=False
    volumes={"/my/project": "/workspace"},
)
# First run: creates container "my-dev-env"
# Next run: finds it, restarts if stopped, reattaches
```

With SessionManager, each user gets their own persistent directory:

```python
manager = SessionManager(
    workspace_root="/app/workspaces",  # Creates /app/workspaces/{user_id}/
)
```

### Custom Sandbox Factory

`SessionManager` accepts a `sandbox_factory` callable to use any sandbox
backend (Daytona, custom implementations, etc.):

```python
from pydantic_ai_backends import SessionManager, DaytonaSandbox

def daytona_factory(session_id: str) -> DaytonaSandbox:
    return DaytonaSandbox(sandbox_id=session_id)

manager = SessionManager(sandbox_factory=daytona_factory)
sandbox = await manager.get_or_create("user-123")
```

When no factory is provided, `SessionManager` defaults to creating
`DockerSandbox` instances (fully backward compatible).

## Security

- Each user gets a separate Docker container
- Users cannot access each other's files
- Containers can have resource limits (CPU, memory)
- Network isolation available via Docker networking
- No host filesystem access (unless explicitly mounted)

## Next Steps

- [Multi-User Example](../examples/multi-user.md) - Web app with SessionManager
- [Docker Sandbox Example](../examples/docker-sandbox.md) - Full example
- [API Reference](../api/docker.md) - Complete API

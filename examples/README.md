# Examples

| Example | What it shows |
|---------|--------------|
| [basic_capability.py](basic_capability.py) | ConsoleCapability with LocalBackend |
| [readonly_agent.py](readonly_agent.py) | Read-only agent with READONLY_RULESET |
| [custom_permissions.py](custom_permissions.py) | Custom ruleset (read + execute, no write) |
| [multi_agent_permissions.py](multi_agent_permissions.py) | Multiple agents with different permissions |

## Running

```bash
export OPENAI_API_KEY=your-key
uv run python examples/basic_capability.py
```

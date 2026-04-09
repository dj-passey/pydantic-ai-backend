"""Docker integration tests for console toolset + permissions (opt-in: pytest -m docker)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_backends import DockerSandbox, create_console_toolset
from pydantic_ai_backends.permissions import (
    OperationPermissions,
    PermissionRule,
    PermissionRuleset,
)


@pytest.fixture(scope="module")
def docker_volume_console(tmp_path_factory):
    pytest.importorskip("docker")
    host = tmp_path_factory.mktemp("console_perm_vol")
    (host / "allowed.txt").write_text("seed\n", encoding="utf-8")
    sandbox = DockerSandbox(volumes={str(host): "/shared"})
    yield sandbox, host
    sandbox.stop()


@dataclass
class _Deps:
    backend: DockerSandbox


def _ctx(deps: _Deps) -> RunContext[_Deps]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


@pytest.mark.docker
class TestConsolePermissionsDockerIntegration:
    @pytest.mark.anyio
    async def test_write_allowed_and_denied_paths(self, docker_volume_console):
        sandbox, _host = docker_volume_console
        rs = PermissionRuleset(
            read=OperationPermissions(default="allow"),
            write=OperationPermissions(
                default="deny",
                rules=[
                    PermissionRule(pattern="**/allowed.txt", action="allow"),
                ],
            ),
            edit=OperationPermissions(
                default="deny",
                rules=[
                    PermissionRule(pattern="**/allowed.txt", action="allow"),
                ],
            ),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(
            permissions=rs,
            include_execute=False,
            require_execute_approval=False,
        )
        deps = _Deps(backend=sandbox)
        c = _ctx(deps)

        denied = await ts.tools["write_file"].function(c, "/shared/other.txt", "nope")
        assert "Error" in denied
        assert "Permission" in denied

        ok = await ts.tools["write_file"].function(c, "/shared/allowed.txt", "updated\n")
        assert "Wrote" in ok

        content = sandbox.read("/shared/allowed.txt")
        assert "updated" in content

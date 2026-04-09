"""Tests for console toolset permission integration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_backends import LocalBackend, StateBackend
from pydantic_ai_backends.permissions import (
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    OperationPermissions,
    PermissionError,
    PermissionRule,
    PermissionRuleset,
)
from pydantic_ai_backends.permissions.checker import PermissionChecker
from pydantic_ai_backends.toolsets.console import (
    _evaluate_toolset_permission,
    _is_denied_by_ruleset,
    _reject_ruleset_per_path_ask,
    _requires_approval_from_ruleset,
    _toolset_permission_prefix_error,
    create_console_toolset,
)


class TestRequiresApprovalFromRuleset:
    """Tests for the _requires_approval_from_ruleset helper."""

    def test_no_ruleset_uses_legacy_flag(self):
        """Test that legacy flag is used when no ruleset."""
        assert _requires_approval_from_ruleset(None, "write", True) is True
        assert _requires_approval_from_ruleset(None, "write", False) is False

    def test_ruleset_with_ask_default(self):
        """Test ruleset with ask as default action."""
        ruleset = PermissionRuleset(
            write=OperationPermissions(default="ask"),
        )
        assert _requires_approval_from_ruleset(ruleset, "write", False) is True

    def test_ruleset_with_allow_default(self):
        """Test ruleset with allow as default action."""
        ruleset = PermissionRuleset(
            write=OperationPermissions(default="allow"),
        )
        assert _requires_approval_from_ruleset(ruleset, "write", True) is False

    def test_ruleset_uses_global_default(self):
        """Test that global default is used for unconfigured operations."""
        ruleset = PermissionRuleset(default="ask")
        assert _requires_approval_from_ruleset(ruleset, "write", False) is True

        ruleset = PermissionRuleset(default="allow")
        assert _requires_approval_from_ruleset(ruleset, "write", True) is False


class TestCreateConsoleToolsetWithPermissions:
    """Tests for create_console_toolset with permissions parameter."""

    def test_create_without_permissions(self):
        """Test creating toolset without permissions."""
        toolset = create_console_toolset()

        # Should use default flags
        assert toolset is not None

    def test_create_with_permissions_ask_write(self):
        """Test creating toolset with permissions that ask for writes."""
        ruleset = PermissionRuleset(
            write=OperationPermissions(default="ask"),
        )
        toolset = create_console_toolset(permissions=ruleset)

        # The toolset should be created successfully
        assert toolset is not None

    def test_create_with_permissions_allow_write(self):
        """Test creating toolset with permissions that allow writes."""
        ruleset = PermissionRuleset(
            write=OperationPermissions(default="allow"),
        )
        toolset = create_console_toolset(permissions=ruleset)

        assert toolset is not None

    def test_create_with_permissions_ask_execute(self):
        """Test creating toolset with permissions that ask for execute."""
        ruleset = PermissionRuleset(
            execute=OperationPermissions(default="ask"),
        )
        toolset = create_console_toolset(permissions=ruleset)

        assert toolset is not None

    def test_create_with_permissions_allow_execute(self):
        """Test creating toolset with permissions that allow execute."""
        ruleset = PermissionRuleset(
            execute=OperationPermissions(default="allow"),
        )
        toolset = create_console_toolset(permissions=ruleset)

        assert toolset is not None

    def test_legacy_flags_ignored_with_permissions(self):
        """Test that legacy flags are ignored when permissions provided."""
        # With permissions, legacy flags should be ignored
        ruleset = PermissionRuleset(
            write=OperationPermissions(default="allow"),
            execute=OperationPermissions(default="allow"),
        )
        toolset = create_console_toolset(
            permissions=ruleset,
            require_write_approval=True,  # Should be ignored
            require_execute_approval=True,  # Should be ignored
        )

        assert toolset is not None

    def test_include_execute_false(self):
        """Test creating toolset without execute tool."""
        ruleset = PermissionRuleset(
            execute=OperationPermissions(default="allow"),
        )
        toolset = create_console_toolset(
            permissions=ruleset,
            include_execute=False,
        )

        assert toolset is not None

    def test_permissions_with_custom_id(self):
        """Test creating toolset with permissions and custom ID."""
        ruleset = PermissionRuleset(default="ask")
        toolset = create_console_toolset(
            id="my-toolset",
            permissions=ruleset,
        )

        assert toolset is not None


class TestIsDeniedByRuleset:
    """Tests for _is_denied_by_ruleset."""

    def test_none_ruleset(self):
        assert _is_denied_by_ruleset(None, "write") is False

    def test_global_default_deny_missing_op(self):
        rs = PermissionRuleset(default="deny")
        assert _is_denied_by_ruleset(rs, "write") is True

    def test_write_unconditional_deny_empty_rules(self):
        rs = PermissionRuleset(
            write=OperationPermissions(default="deny", rules=[]),
        )
        assert _is_denied_by_ruleset(rs, "write") is True

    def test_write_unconditional_deny_only_deny_rules(self):
        rs = PermissionRuleset(
            write=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/.env", action="deny")],
            ),
        )
        assert _is_denied_by_ruleset(rs, "write") is True

    def test_write_default_deny_with_allow_rule_not_stripped(self):
        rs = PermissionRuleset(
            write=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/ok.txt", action="allow")],
            ),
        )
        assert _is_denied_by_ruleset(rs, "write") is False

    def test_write_default_allow(self):
        rs = PermissionRuleset(
            write=OperationPermissions(default="allow"),
        )
        assert _is_denied_by_ruleset(rs, "write") is False


class TestRejectRulesetPerPathAsk:
    """Rulesets with per-path ask (rule action ask, default != ask) must fail."""

    def test_allow_default_with_ask_rule_raises(self):
        rs = PermissionRuleset(
            write=OperationPermissions(
                default="allow",
                rules=[PermissionRule(pattern="**/x", action="ask")],
            ),
        )
        with pytest.raises(NotImplementedError, match="Per-path 'ask'"):
            _reject_ruleset_per_path_ask(rs)

    def test_deny_default_with_ask_rule_raises(self):
        rs = PermissionRuleset(
            write=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/x", action="ask")],
            ),
        )
        with pytest.raises(NotImplementedError, match="Per-path 'ask'"):
            _reject_ruleset_per_path_ask(rs)

    def test_ask_default_with_ask_rule_ok(self):
        rs = PermissionRuleset(
            write=OperationPermissions(
                default="ask",
                rules=[PermissionRule(pattern="**/x", action="ask")],
            ),
        )
        _reject_ruleset_per_path_ask(rs)

    def test_create_toolset_raises_for_per_path_ask(self):
        rs = PermissionRuleset(
            read=OperationPermissions(
                default="allow",
                rules=[PermissionRule(pattern="**/s", action="ask")],
            ),
        )
        with pytest.raises(NotImplementedError):
            create_console_toolset(permissions=rs)


class TestEvaluateToolsetPermission:
    """Direct tests for _evaluate_toolset_permission branches."""

    def test_skips_when_backend_has_permission_checker(self):
        checker = PermissionChecker(ruleset=PermissionRuleset(default="deny"))

        class BackendWithChecker:
            permission_checker = object()

        assert _evaluate_toolset_permission(checker, BackendWithChecker(), "read", "/x") is None

    def test_allow_action(self):
        checker = PermissionChecker(
            ruleset=PermissionRuleset(
                read=OperationPermissions(default="allow"),
            )
        )

        class NoChecker:
            pass

        assert _evaluate_toolset_permission(checker, NoChecker(), "read", "/a") is None

    def test_deny_with_rule_description(self):
        checker = PermissionChecker(
            ruleset=PermissionRuleset(
                read=OperationPermissions(
                    default="allow",
                    rules=[
                        PermissionRule(
                            pattern="**/blocked.txt",
                            action="deny",
                            description="no way",
                        )
                    ],
                )
            )
        )

        class NoChecker:
            pass

        msg = _evaluate_toolset_permission(checker, NoChecker(), "read", "/tmp/blocked.txt")
        assert msg is not None
        assert "no way" in msg

    def test_deny_without_description(self):
        checker = PermissionChecker(
            ruleset=PermissionRuleset(
                read=OperationPermissions(
                    default="allow",
                    rules=[
                        PermissionRule(pattern="**/z.txt", action="deny"),
                    ],
                )
            )
        )

        class NoChecker:
            pass

        msg = _evaluate_toolset_permission(checker, NoChecker(), "read", "/z.txt")
        assert msg is not None
        assert "Permission denied for read" in msg

    def test_ask_without_callback_raises(self):
        checker = PermissionChecker(ruleset=PermissionRuleset(default="ask"))

        class NoChecker:
            pass

        with pytest.raises(PermissionError):
            _evaluate_toolset_permission(checker, NoChecker(), "read", "/any")


class TestToolsetPermissionPrefixError:
    def test_none_checker(self):
        class NoChecker:
            pass

        assert _toolset_permission_prefix_error(None, NoChecker(), "read", "/x") is None

    def test_wraps_permission_error(self):
        checker = PermissionChecker(ruleset=PermissionRuleset(default="ask"))

        class NoChecker:
            pass

        out = _toolset_permission_prefix_error(checker, NoChecker(), "read", "/p")
        assert out is not None
        assert out.startswith("Error: Permission required")


@dataclass
class _MockDeps:
    backend: LocalBackend | StateBackend


def _run_ctx(deps: _MockDeps) -> RunContext[_MockDeps]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


class TestToolRegistrationWithAllowRules:
    def test_write_tools_present_when_default_deny_with_allow_rule(self):
        rs = PermissionRuleset(
            default="deny",
            read=OperationPermissions(default="allow"),
            write=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/w.txt", action="allow")],
            ),
            edit=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/w.txt", action="allow")],
            ),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(permissions=rs, include_execute=False)
        assert "write_file" in ts.tools
        assert "edit_file" in ts.tools
        assert "execute" not in ts.tools


class TestConsoleToolsetEnforcementStateBackend:
    @pytest.mark.anyio
    async def test_write_denied_outside_allow_list(self):
        backend = StateBackend()
        rs = PermissionRuleset(
            read=OperationPermissions(default="allow"),
            write=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/ok.txt", action="allow")],
            ),
            edit=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/ok.txt", action="allow")],
            ),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(permissions=rs, include_execute=False)
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        bad = await ts.tools["write_file"].function(c, "bad.txt", "x")
        assert "Error" in bad
        assert "Permission" in bad

    @pytest.mark.anyio
    async def test_write_allowed_for_matched_path(self):
        backend = StateBackend()
        rs = PermissionRuleset(
            read=OperationPermissions(default="allow"),
            write=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/ok.txt", action="allow")],
            ),
            edit=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/ok.txt", action="allow")],
            ),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(permissions=rs, include_execute=False)
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        ok = await ts.tools["write_file"].function(c, "ok.txt", "line\n")
        assert ok.startswith("Wrote")

    @pytest.mark.anyio
    async def test_read_denied(self):
        backend = StateBackend()
        backend.write("/x.txt", "a")
        rs = PermissionRuleset(
            read=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/y.txt", action="allow")],
            ),
            write=OperationPermissions(default="allow"),
            edit=OperationPermissions(default="allow"),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(permissions=rs, include_execute=False)
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        out = await ts.tools["read_file"].function(c, "/x.txt")
        assert "Error" in out

    @pytest.mark.anyio
    async def test_ls_glob_grep_execute(self, tmp_path):
        backend = LocalBackend(root_dir=tmp_path, enable_execute=True)
        (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
        rs = PermissionRuleset(
            read=OperationPermissions(default="allow"),
            write=OperationPermissions(default="allow"),
            edit=OperationPermissions(default="allow"),
            execute=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="echo*", action="allow")],
            ),
            ls=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern=".", action="allow")],
            ),
            glob=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern=".", action="allow")],
            ),
            grep=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern=".", action="allow")],
            ),
        )
        ts = create_console_toolset(permissions=rs, require_execute_approval=False)
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)

        ls_bad = await ts.tools["ls"].function(c, "..")
        assert "Error" in ls_bad

        ls_ok = await ts.tools["ls"].function(c, ".")
        assert "Contents" in ls_ok

        glob_bad = await ts.tools["glob"].function(c, "*.txt", "..")
        assert "Error" in glob_bad

        glob_ok = await ts.tools["glob"].function(c, "*.txt", ".")
        assert "Found" in glob_ok

        grep_bad = await ts.tools["grep"].function(c, "hi", path="..")
        assert "Error" in grep_bad

        grep_ok = await ts.tools["grep"].function(c, "hi", path=".")
        assert "hi" in grep_ok or "Files containing" in grep_ok

        ex_bad = await ts.tools["execute"].function(c, "rm -f a.txt", timeout=5)
        assert "Error" in ex_bad

        ex_ok = await ts.tools["execute"].function(c, "echo ok", timeout=5)
        assert "ok" in ex_ok


class TestBackendPermissionCheckerWins:
    @pytest.mark.anyio
    async def test_local_backend_readonly_blocks_write_despite_permissive_toolset(self, tmp_path):
        (tmp_path / "f.txt").write_text("a", encoding="utf-8")
        backend = LocalBackend(root_dir=tmp_path, permissions=READONLY_RULESET)
        ts = create_console_toolset(
            permissions=PERMISSIVE_RULESET,
            require_execute_approval=False,
        )
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        out = await ts.tools["write_file"].function(c, "f.txt", "b")
        assert "Error" in out


class TestHashlineFormatPermissions:
    @pytest.mark.anyio
    async def test_read_denied_hashline_branch(self):
        backend = StateBackend()
        backend.write("/f.txt", "x\n")
        rs = PermissionRuleset(
            read=OperationPermissions(default="deny"),
            write=OperationPermissions(default="allow"),
            edit=OperationPermissions(default="allow"),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(
            edit_format="hashline",
            permissions=rs,
            include_execute=False,
        )
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        out = await ts.tools["read_file"].function(c, "/f.txt")
        assert "Error" in out

    @pytest.mark.anyio
    async def test_hashline_edit_respects_edit_permission(self):
        from pydantic_ai_backends.hashline import line_hash

        backend = StateBackend()
        backend.write("/w.txt", "one\n")
        rs = PermissionRuleset(
            read=OperationPermissions(default="allow"),
            write=OperationPermissions(default="allow"),
            edit=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/other.txt", action="allow")],
            ),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(
            edit_format="hashline",
            permissions=rs,
            include_execute=False,
        )
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        h = line_hash("one")
        bad = await ts.tools["hashline_edit"].function(c, "/w.txt", 1, h, "two", None, None, False)
        assert "Error" in bad
        assert "Permission" in bad

    @pytest.mark.anyio
    async def test_hashline_edit_allowed(self):
        from pydantic_ai_backends.hashline import line_hash

        backend = StateBackend()
        backend.write("/w.txt", "one\n")
        rs = PermissionRuleset(
            read=OperationPermissions(default="allow"),
            write=OperationPermissions(default="allow"),
            edit=OperationPermissions(
                default="deny",
                rules=[PermissionRule(pattern="**/w.txt", action="allow")],
            ),
            execute=OperationPermissions(default="deny"),
            ls=OperationPermissions(default="allow"),
            glob=OperationPermissions(default="allow"),
            grep=OperationPermissions(default="allow"),
        )
        ts = create_console_toolset(
            edit_format="hashline",
            permissions=rs,
            include_execute=False,
        )
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        h = line_hash("one")
        ok = await ts.tools["hashline_edit"].function(c, "/w.txt", 1, h, "two", None, None, False)
        assert ok.startswith("Edited")


class TestAskOnToolsetWithoutCallback:
    @pytest.mark.anyio
    async def test_global_ask_returns_error_string(self):
        backend = StateBackend()
        rs = PermissionRuleset(default="ask")
        ts = create_console_toolset(
            permissions=rs,
            include_execute=False,
        )
        deps = _MockDeps(backend=backend)
        c = _run_ctx(deps)
        out = await ts.tools["read_file"].function(c, "/nope.txt")
        assert "Error" in out
        assert "Permission required" in out

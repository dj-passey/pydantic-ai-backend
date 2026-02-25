# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.12] - 2026-02-25

### Added

- **`DaytonaSandbox` — cloud sandbox backend** powered by [Daytona](https://daytona.io/) ephemeral sandboxes. Sub-90ms startup, no Docker daemon required. Install with `pip install pydantic-ai-backend[daytona]`.
  - Native Daytona file APIs for efficient read/write
  - Auth via `DAYTONA_API_KEY` env var or `api_key=` parameter
  - New `[daytona]` optional dependency group
- **Documentation**: New [Daytona Sandbox](concepts/daytona.md) concept guide

### Changed

- **Extracted `BaseSandbox` to `backends/base.py`** — no longer nested in Docker subpackage. All existing import paths remain backward compatible.

## [0.1.4] - 2025-01-22

### Changed

- **README**: Complete rewrite with centered header, badges, Use Cases table, and vstorm-co branding
- **Documentation**: Updated styling to match pydantic-deep pink theme
  - Inter font for text, JetBrains Mono for code
  - Pink accent color scheme
  - Custom CSS and announcement bar
- **mkdocs.yml**: Updated with full Material theme configuration

### Added

- **Custom Styling**: docs/overrides/main.html, docs/stylesheets/extra.css
- **Abbreviations**: docs/includes/abbreviations.md for markdown expansions
- **FAQ Section**: Expanded getting-help.md with common questions

## [0.1.3] - 2026-01-22

### Fixed

- `DockerSandbox.edit()` now handles multiline strings correctly. Replaced sed/grep-based
  implementation with Python string operations, which naturally handle newlines and special
  characters without shell escaping issues. ([#6](https://github.com/vstorm-co/pydantic-ai-backend/pull/6))

### Changed

- Added `edit()` as an abstract method in `BaseSandbox` to make the interface explicit
- Docker tests now use shared fixtures (`scope="module"`) for faster test execution

## [0.1.2] - 2026-01-21

### Added

- **Fine-grained Permission System** - Pattern-based access control for file operations and shell execution
  - `PermissionRuleset` - Complete permission configuration with per-operation rules
  - `PermissionRule` - Glob pattern matching with `allow`, `deny`, or `ask` actions
  - `PermissionChecker` - Validates operations against rulesets with async callback support
  - `OperationPermissions` - Per-operation default actions and override rules

- **Pre-configured Permission Presets**
  - `DEFAULT_RULESET` - Safe defaults (allow reads except secrets, ask for writes/executes)
  - `PERMISSIVE_RULESET` - Allow most operations, deny only dangerous commands
  - `READONLY_RULESET` - Allow read operations only, deny all writes and executes
  - `STRICT_RULESET` - Everything requires explicit approval

- **Permission Integration**
  - `permissions` parameter in `LocalBackend` for fine-grained access control
  - `permissions` parameter in `create_console_toolset()` for tool approval requirements
  - `ask_callback` parameter for interactive permission prompts
  - `ask_fallback` parameter to control behavior when callback unavailable ("deny" or "error")

- **Helper Functions and Patterns**
  - `create_ruleset()` factory function for custom permission configurations
  - `SECRETS_PATTERNS` - Common patterns for sensitive files (.env, .pem, credentials, etc.)
  - `SYSTEM_PATTERNS` - Common patterns for system directories (/etc, /var, etc.)
  - `is_allowed()`, `is_denied()`, `requires_approval()` convenience methods

- **New Exceptions**
  - `PermissionError` - Raised when approval required but no callback available
  - `PermissionDeniedError` - Raised when operation is explicitly denied

### Changed

- `LocalBackend` now checks permissions after `allowed_directories` validation
- Legacy `require_write_approval` and `require_execute_approval` flags are preserved for backward compatibility but `permissions` parameter takes precedence when provided

### Fixed

- `DockerSandbox.execute()` no longer incorrectly escapes commands when timeout is specified.
  Previously, using Python's `repr()` for shell quoting caused issues with commands containing
  mixed quotes or special characters. ([#3](https://github.com/vstorm-co/pydantic-ai-backend/pull/3))

### Documentation

- New "Permissions" concept guide with examples
- Updated backends documentation with permission examples
- Updated console toolset documentation with permission configuration
- API reference for all permission types and functions

## [0.1.1] - 2026-01-20

### Added

- `ignore_hidden` parameter to `grep_raw()` in `BackendProtocol` - controls whether hidden
  files (dotfiles) are included in search results (default: `True` = ignore hidden files).
- `default_ignore_hidden` option in `create_console_toolset()` so agents can opt-in to
  searching dotfiles by default.
- `--include-hidden` CLI flag in the `examples/local_cli` agent to expose the new grep
  behavior from the command line.

### Changed

- `grep_raw()` now consistently ignores hidden files by default across all backends
  (`StateBackend`, `LocalBackend`, `CompositeBackend`, `DockerSandbox`).
- Console `grep` tool now treats `ignore_hidden` as an override parameter, falling back to
  the toolset's default when omitted.

## [0.1.0] - 2025-01-17

### Added

- **`LocalBackend`** - Unified backend for local filesystem + shell execution
- **Console Toolset** - Ready-to-use pydantic-ai toolset
- **Full Documentation** - MkDocs Material theme documentation
- New `[console]` optional dependency for pydantic-ai toolset
- Real Coveralls integration for dynamic coverage badge
- Architecture diagram in README

### Changed

- **Project Structure Reorganization**
- Updated all imports to new module paths
- Version now loaded dynamically from package metadata
- README rewritten with pydantic-ai agent examples

### Removed

- **`FilesystemBackend`** - Replaced by `LocalBackend`
- **`LocalSandbox`** - Replaced by `LocalBackend`

## [0.0.4] - 2025-01-16

### Added

- `volumes` parameter to `DockerSandbox` for mounting host directories
- `workspace_root` parameter to `SessionManager` for automatic per-session storage

## [0.0.1] - 2025-12-28

### Added

- Initial release extracted from pydantic-deep
- `BackendProtocol` - Unified interface for file operations
- `SandboxProtocol` - Extended interface for command execution
- `StateBackend` - In-memory file storage
- `DockerSandbox` - Docker container-based sandbox
- `SessionManager` - Multi-user session management
- Built-in runtimes: python-minimal, python-datascience, python-web, node-minimal, node-react

[0.1.4]: https://github.com/vstorm-co/pydantic-ai-backend/releases/tag/v0.1.4
[0.1.3]: https://github.com/vstorm-co/pydantic-ai-backend/releases/tag/v0.1.3
[0.1.2]: https://github.com/vstorm-co/pydantic-ai-backend/releases/tag/v0.1.2
[0.1.1]: https://github.com/vstorm-co/pydantic-ai-backend/releases/tag/v0.1.1
[0.1.0]: https://github.com/vstorm-co/pydantic-ai-backend/releases/tag/v0.1.0
[0.0.4]: https://github.com/vstorm-co/pydantic-ai-backend/releases/tag/v0.0.4
[0.0.1]: https://github.com/vstorm-co/pydantic-ai-backend/releases/tag/v0.0.1

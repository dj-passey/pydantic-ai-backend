"""Tests for hashline module — content-hash-tagged line editing."""

from pydantic_ai_backends.hashline import (
    _split_lines,
    apply_hashline_edit,
    apply_hashline_edit_with_summary,
    format_hashline_output,
    line_hash,
)


class TestLineHash:
    """Test line_hash function."""

    def test_returns_two_char_hex(self):
        h = line_hash("hello")
        assert len(h) == 2
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert line_hash("foo") == line_hash("foo")

    def test_different_content_different_hash(self):
        # Not guaranteed for all inputs, but these are different enough
        h1 = line_hash("def hello():")
        h2 = line_hash("def world():")
        # Could collide in theory, but extremely unlikely for these
        assert h1 != h2

    def test_empty_string(self):
        h = line_hash("")
        assert len(h) == 2

    def test_whitespace_matters(self):
        h1 = line_hash("  hello")
        h2 = line_hash("hello")
        assert h1 != h2


class TestSplitLines:
    """Test _split_lines helper."""

    def test_no_trailing_newline(self):
        lines, has_nl = _split_lines("a\nb\nc")
        assert lines == ["a", "b", "c"]
        assert has_nl is False

    def test_with_trailing_newline(self):
        lines, has_nl = _split_lines("a\nb\nc\n")
        assert lines == ["a", "b", "c"]
        assert has_nl is True

    def test_empty_string(self):
        lines, has_nl = _split_lines("")
        assert lines == [""]
        assert has_nl is False

    def test_single_newline(self):
        lines, has_nl = _split_lines("\n")
        assert lines == [""]
        assert has_nl is True

    def test_multiple_empty_lines(self):
        lines, has_nl = _split_lines("a\n\nb\n")
        assert lines == ["a", "", "b"]
        assert has_nl is True


class TestFormatHashlineOutput:
    """Test format_hashline_output function."""

    def test_basic_format(self):
        content = "hello\nworld\n"
        result = format_hashline_output(content)
        lines = result.split("\n")
        assert len(lines) == 2
        # Check format: {num}:{hash}|{content}
        assert lines[0].startswith("1:")
        assert "|hello" in lines[0]
        assert lines[1].startswith("2:")
        assert "|world" in lines[1]

    def test_hash_format(self):
        result = format_hashline_output("test line\n")
        # Should be "1:XX|test line" where XX is 2-char hex
        parts = result.split("|", 1)
        prefix = parts[0]  # "1:XX"
        assert ":" in prefix
        num, h = prefix.split(":")
        assert num == "1"
        assert len(h) == 2

    def test_offset(self):
        content = "line1\nline2\nline3\nline4\n"
        result = format_hashline_output(content, offset=2)
        lines = result.split("\n")
        assert lines[0].startswith("3:")  # 0-indexed offset 2 → line 3
        assert "|line3" in lines[0]

    def test_limit(self):
        content = "a\nb\nc\nd\ne\n"
        result = format_hashline_output(content, limit=2)
        lines = result.strip().split("\n")
        # 2 content lines + blank + truncation message
        assert "|a" in lines[0]
        assert "|b" in lines[1]
        assert "3 more lines" in result

    def test_offset_and_limit(self):
        content = "a\nb\nc\nd\ne\n"
        result = format_hashline_output(content, offset=1, limit=2)
        lines = result.strip().split("\n")
        assert "|b" in lines[0]
        assert "|c" in lines[1]
        assert "2 more lines" in result

    def test_empty_file(self):
        # An empty string splits to [""] — one empty line
        result = format_hashline_output("")
        assert result.startswith("1:")
        assert "|" in result

    def test_truly_empty_file(self):
        # A file that is just a newline → _split_lines removes trailing empty
        # giving 1 empty line; but "\n" → ["", ""] → [""] after strip
        result = format_hashline_output("\n")
        assert result.startswith("1:")

    def test_offset_exceeds_length(self):
        result = format_hashline_output("hello\n", offset=10)
        assert "Error" in result
        assert "exceeds" in result

    def test_no_trailing_newline(self):
        content = "hello\nworld"
        result = format_hashline_output(content)
        lines = result.split("\n")
        assert len(lines) == 2

    def test_single_line(self):
        result = format_hashline_output("only line\n")
        assert result.startswith("1:")
        assert "|only line" in result

    def test_hashes_match_line_hash(self):
        content = "foo\nbar\nbaz\n"
        result = format_hashline_output(content)
        for output_line in result.split("\n"):
            prefix, text = output_line.split("|", 1)
            _, h = prefix.split(":")
            assert h == line_hash(text)


class TestApplyHashlineEdit:
    """Test apply_hashline_edit function."""

    def _content_and_hash(self, content: str, line_num: int) -> str:
        """Get hash for a specific line number (1-indexed)."""
        lines = content.split("\n")
        if content.endswith("\n"):
            lines = lines[:-1]
        return line_hash(lines[line_num - 1])

    def test_replace_single_line(self):
        content = "aaa\nbbb\nccc\n"
        h = self._content_and_hash(content, 2)
        new, error = apply_hashline_edit(content, 2, h, "BBB")
        assert error is None
        assert new == "aaa\nBBB\nccc\n"

    def test_replace_range(self):
        content = "aaa\nbbb\nccc\nddd\n"
        h1 = self._content_and_hash(content, 2)
        h2 = self._content_and_hash(content, 3)
        new, error = apply_hashline_edit(content, 2, h1, "XXX\nYYY", end_line=3, end_hash=h2)
        assert error is None
        assert new == "aaa\nXXX\nYYY\nddd\n"

    def test_insert_after(self):
        content = "aaa\nbbb\nccc\n"
        h = self._content_and_hash(content, 2)
        new, error = apply_hashline_edit(content, 2, h, "NEW", insert_after=True)
        assert error is None
        assert new == "aaa\nbbb\nNEW\nccc\n"

    def test_delete_single_line(self):
        content = "aaa\nbbb\nccc\n"
        h = self._content_and_hash(content, 2)
        new, error = apply_hashline_edit(content, 2, h, "")
        assert error is None
        assert new == "aaa\nccc\n"

    def test_delete_range(self):
        content = "aaa\nbbb\nccc\nddd\n"
        h1 = self._content_and_hash(content, 2)
        h2 = self._content_and_hash(content, 3)
        new, error = apply_hashline_edit(content, 2, h1, "", end_line=3, end_hash=h2)
        assert error is None
        assert new == "aaa\nddd\n"

    def test_hash_mismatch_start(self):
        content = "aaa\nbbb\nccc\n"
        new, error = apply_hashline_edit(content, 2, "zz", "XXX")
        assert error is not None
        assert "Hash mismatch" in error
        assert "line 2" in error
        assert new == content  # unchanged

    def test_hash_mismatch_end(self):
        content = "aaa\nbbb\nccc\n"
        h1 = self._content_and_hash(content, 1)
        new, error = apply_hashline_edit(content, 1, h1, "XXX", end_line=3, end_hash="zz")
        assert error is not None
        assert "Hash mismatch" in error
        assert "line 3" in error

    def test_start_line_out_of_range_zero(self):
        content = "aaa\n"
        new, error = apply_hashline_edit(content, 0, "aa", "XXX")
        assert error is not None
        assert "out of range" in error

    def test_start_line_out_of_range_too_large(self):
        content = "aaa\n"
        new, error = apply_hashline_edit(content, 5, "aa", "XXX")
        assert error is not None
        assert "out of range" in error

    def test_end_line_before_start(self):
        content = "aaa\nbbb\nccc\n"
        h = self._content_and_hash(content, 2)
        new, error = apply_hashline_edit(content, 2, h, "XXX", end_line=1)
        assert error is not None
        assert "must be >=" in error

    def test_end_line_out_of_range(self):
        content = "aaa\nbbb\n"
        h = self._content_and_hash(content, 1)
        new, error = apply_hashline_edit(content, 1, h, "XXX", end_line=10)
        assert error is not None
        assert "out of range" in error

    def test_preserves_trailing_newline(self):
        content = "aaa\nbbb\n"
        h = self._content_and_hash(content, 1)
        new, error = apply_hashline_edit(content, 1, h, "XXX")
        assert error is None
        assert new.endswith("\n")

    def test_no_trailing_newline_preserved(self):
        content = "aaa\nbbb"
        h = self._content_and_hash(content, 1)
        new, error = apply_hashline_edit(content, 1, h, "XXX")
        assert error is None
        assert not new.endswith("\n")

    def test_replace_first_line(self):
        content = "old\nkeep\n"
        h = self._content_and_hash(content, 1)
        new, error = apply_hashline_edit(content, 1, h, "new")
        assert error is None
        assert new == "new\nkeep\n"

    def test_replace_last_line(self):
        content = "keep\nold\n"
        h = self._content_and_hash(content, 2)
        new, error = apply_hashline_edit(content, 2, h, "new")
        assert error is None
        assert new == "keep\nnew\n"

    def test_insert_multiline(self):
        content = "aaa\nbbb\n"
        h = self._content_and_hash(content, 1)
        new, error = apply_hashline_edit(content, 1, h, "x\ny\nz", insert_after=True)
        assert error is None
        assert new == "aaa\nx\ny\nz\nbbb\n"

    def test_end_hash_optional(self):
        content = "aaa\nbbb\nccc\n"
        h = self._content_and_hash(content, 1)
        new, error = apply_hashline_edit(content, 1, h, "XXX", end_line=2)
        assert error is None
        assert new == "XXX\nccc\n"

    def test_replace_all_lines(self):
        content = "aaa\nbbb\nccc\n"
        h1 = self._content_and_hash(content, 1)
        h3 = self._content_and_hash(content, 3)
        new, error = apply_hashline_edit(content, 1, h1, "new", end_line=3, end_hash=h3)
        assert error is None
        assert new == "new\n"


class TestApplyHashlineEditWithSummary:
    """Test apply_hashline_edit_with_summary function."""

    def _hash(self, content: str, line_num: int) -> str:
        lines = content.split("\n")
        if content.endswith("\n"):
            lines = lines[:-1]
        return line_hash(lines[line_num - 1])

    def test_replace_summary(self):
        content = "aaa\nbbb\nccc\n"
        h = self._hash(content, 2)
        _, error, summary = apply_hashline_edit_with_summary(content, 2, h, "XXX")
        assert error is None
        assert "Replaced 1 line(s)" in summary

    def test_insert_summary(self):
        content = "aaa\nbbb\n"
        h = self._hash(content, 1)
        _, error, summary = apply_hashline_edit_with_summary(
            content, 1, h, "x\ny", insert_after=True
        )
        assert error is None
        assert "Inserted 2 line(s)" in summary

    def test_delete_summary(self):
        content = "aaa\nbbb\nccc\n"
        h = self._hash(content, 2)
        _, error, summary = apply_hashline_edit_with_summary(content, 2, h, "")
        assert error is None
        assert "Deleted 1 line(s)" in summary

    def test_replace_range_different_count_summary(self):
        content = "aaa\nbbb\nccc\nddd\n"
        h1 = self._hash(content, 2)
        h3 = self._hash(content, 3)
        _, error, summary = apply_hashline_edit_with_summary(
            content, 2, h1, "X", end_line=3, end_hash=h3
        )
        assert error is None
        assert "Replaced 2 line(s) with 1 line(s)" in summary

    def test_error_returns_empty_summary(self):
        content = "aaa\n"
        _, error, summary = apply_hashline_edit_with_summary(content, 5, "zz", "X")
        assert error is not None
        assert summary == ""

    def test_hash_mismatch_start_returns_empty_summary(self):
        content = "aaa\n"
        _, error, summary = apply_hashline_edit_with_summary(content, 1, "zz", "X")
        assert error is not None
        assert "Hash mismatch" in error
        assert summary == ""

    def test_end_before_start_returns_empty_summary(self):
        content = "aaa\nbbb\n"
        h = self._hash(content, 2)
        _, error, summary = apply_hashline_edit_with_summary(content, 2, h, "X", end_line=1)
        assert error is not None
        assert summary == ""

    def test_end_out_of_range_returns_empty_summary(self):
        content = "aaa\n"
        h = self._hash(content, 1)
        _, error, summary = apply_hashline_edit_with_summary(content, 1, h, "X", end_line=10)
        assert error is not None
        assert summary == ""

    def test_end_hash_mismatch_returns_empty_summary(self):
        content = "aaa\nbbb\n"
        h = self._hash(content, 1)
        _, error, summary = apply_hashline_edit_with_summary(
            content, 1, h, "X", end_line=2, end_hash="zz"
        )
        assert error is not None
        assert summary == ""


class TestConsoleToolsetHashlineFormat:
    """Test create_console_toolset with edit_format='hashline'."""

    def test_hashline_toolset_has_hashline_edit(self):
        from pydantic_ai_backends import create_console_toolset

        toolset = create_console_toolset(edit_format="hashline")
        tool_names = list(toolset.tools.keys())
        assert "hashline_edit" in tool_names
        assert "edit_file" not in tool_names

    def test_hashline_toolset_has_read_file(self):
        from pydantic_ai_backends import create_console_toolset

        toolset = create_console_toolset(edit_format="hashline")
        assert "read_file" in toolset.tools

    def test_hashline_toolset_has_standard_tools(self):
        from pydantic_ai_backends import create_console_toolset

        toolset = create_console_toolset(edit_format="hashline")
        tool_names = list(toolset.tools.keys())
        assert "ls" in tool_names
        assert "write_file" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names

    def test_str_replace_toolset_has_edit_file(self):
        from pydantic_ai_backends import create_console_toolset

        toolset = create_console_toolset(edit_format="str_replace")
        tool_names = list(toolset.tools.keys())
        assert "edit_file" in tool_names
        assert "hashline_edit" not in tool_names

    def test_hashline_system_prompt(self):
        from pydantic_ai_backends import get_console_system_prompt

        prompt = get_console_system_prompt(edit_format="hashline")
        assert "hashline" in prompt.lower()
        assert "hash" in prompt.lower()
        assert "hashline_edit" in prompt

    def test_str_replace_system_prompt(self):
        from pydantic_ai_backends import get_console_system_prompt

        prompt = get_console_system_prompt(edit_format="str_replace")
        assert "edit_file" in prompt
        assert "hashline" not in prompt.lower()

    def test_default_system_prompt_is_str_replace(self):
        from pydantic_ai_backends import get_console_system_prompt

        prompt = get_console_system_prompt()
        assert "edit_file" in prompt

    def test_hashline_edit_requires_write_approval(self):
        from pydantic_ai_backends import create_console_toolset

        toolset = create_console_toolset(edit_format="hashline", require_write_approval=True)
        assert toolset.tools["hashline_edit"].requires_approval is True
        assert toolset.tools["write_file"].requires_approval is True


class TestLazyImports:
    """Test lazy imports for hashline module."""

    def test_import_line_hash(self):
        from pydantic_ai_backends import line_hash

        assert callable(line_hash)

    def test_import_format_hashline_output(self):
        from pydantic_ai_backends import format_hashline_output

        assert callable(format_hashline_output)

    def test_import_apply_hashline_edit(self):
        from pydantic_ai_backends import apply_hashline_edit

        assert callable(apply_hashline_edit)

    def test_import_apply_hashline_edit_with_summary(self):
        from pydantic_ai_backends import apply_hashline_edit_with_summary

        assert callable(apply_hashline_edit_with_summary)

    def test_import_edit_format(self):
        from pydantic_ai_backends import EditFormat

        assert EditFormat is not None

    def test_import_hashline_console_prompt(self):
        from pydantic_ai_backends import HASHLINE_CONSOLE_PROMPT

        assert isinstance(HASHLINE_CONSOLE_PROMPT, str)
        assert "hashline" in HASHLINE_CONSOLE_PROMPT.lower()

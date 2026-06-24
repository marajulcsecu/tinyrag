"""Tests for src/tinyrag/observability/logger.py — the structured logger.

Test layout
-----------
- TestPublicSurface     — module exports ``get_logger`` and
  ``configure_logging``; subpackage ``tinyrag.observability``
  re-exports them.
- TestConfigureLogging  — the two-pipeline dictConfig is built
  correctly (handlers, formatters, levels). Idempotent: calling
  configure_logging twice doesn't crash.
- TestGetLogger         — returns a bound logger that responds to
  the standard structlog methods (info, warning, error, debug).
- TestStdoutPretty      — default ``json_format: false`` →
  stdout gets pretty (non-JSON) output.
- TestStdoutJson        — ``json_format: true`` → stdout gets
  JSON output.
- TestFileJson          — file handler emits JSON regardless of
  ``json_format``. Confirms the architecture doc §12.1 invariant:
  "logs/tinyrag.log (JSON, append-only, for postmortem)".
- TestFileDisabled      — ``path: None`` → no file handler, no
  file is created.
- TestFileDirCreated    — missing log-file parent directory is
  auto-created.
- TestFileDirError      — unwritable parent directory raises
  LoggingError (clean error, not a stack trace).
- TestLogEventFormat    — emitted events have all required fields
  (timestamp, level, logger, event).
- TestThirdPartyQuieted — httpx / httpcore / sentence_transformers
  are at WARNING level by default.

Why so many tests?
------------------
Logging is hard to debug because a broken config only shows up
when someone runs the program and looks at the output. These
tests catch the most common failure modes (missing formatter,
wrong handler, file in the wrong place, etc.) without requiring
a developer to actually run TinyRAG end-to-end.

Location: ``tests/test_logger.py``
"""

from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
from pathlib import Path

import pytest
import structlog

from tinyrag.config import LoggingSettings, LogLevel
from tinyrag.observability import LoggingError, configure_logging, get_logger
from tinyrag.observability.logger import _build_dict_config

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def reset_logging():
    """Snapshot and restore the global logging state around each test.

    ``dictConfig`` mutates the root logger + handler registry, and
    structlog's ``configure`` is also global. Without this fixture
    one test's handlers would leak into the next, producing
    confusing "extra output" failures.
    """
    # Snapshot root logger.
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_disable_existing = logging.root.manager.disable

    yield

    # Restore: remove any handlers dictConfig added, restore snapshot.
    for h in list(root.handlers):
        with contextlib.suppress(Exception):
            h.close()
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)
    logging.root.manager.disable = saved_disable_existing

    # Reset structlog so the next test gets a fresh configure call.
    structlog.reset_defaults()


@pytest.fixture
def fresh_settings(tmp_path: Path) -> LoggingSettings:
    """A LoggingSettings with a tmp log path that won't pollute the repo."""
    return LoggingSettings(
        level=LogLevel.INFO,
        json_format=False,
        path=str(tmp_path / "subdir" / "tinyrag.log"),
    )


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestPublicSurface:
    """The expected symbols are exported and importable."""

    def test_subpackage_exports_configure_logging(self) -> None:
        from tinyrag.observability import configure_logging as fn

        assert fn is configure_logging

    def test_subpackage_exports_get_logger(self) -> None:
        from tinyrag.observability import get_logger as fn

        assert fn is get_logger

    def test_subpackage_exports_logging_error(self) -> None:
        from tinyrag.observability import LoggingError as cls

        assert cls is LoggingError

    def test_get_logger_returns_bound_logger(self) -> None:
        """get_logger(name) returns a structlog BoundLogger."""
        log = get_logger("my.module")
        # structlog.stdlib.BoundLogger is the post-configure type.
        # Even without configure_logging() being called, structlog
        # has a default factory that produces a usable logger.
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "debug")


class TestBuildDictConfig:
    """The internal _build_dict_config helper produces a valid dictConfig."""

    def test_handlers_include_stdout(self, fresh_settings: LoggingSettings) -> None:
        cfg = _build_dict_config(fresh_settings, None)
        assert "stdout" in cfg["handlers"]

    def test_handlers_exclude_file_when_path_is_none(
        self, fresh_settings: LoggingSettings
    ) -> None:
        cfg = _build_dict_config(fresh_settings, None)
        assert "file" not in cfg["handlers"]

    def test_handlers_include_file_when_path_set(
        self, fresh_settings: LoggingSettings
    ) -> None:
        cfg = _build_dict_config(fresh_settings, Path("/tmp/some.log"))
        assert "file" in cfg["handlers"]
        assert cfg["handlers"]["file"]["filename"] == "/tmp/some.log"

    def test_stdout_formatter_is_pretty_by_default(
        self, fresh_settings: LoggingSettings
    ) -> None:
        cfg = _build_dict_config(fresh_settings, None)
        assert cfg["handlers"]["stdout"]["formatter"] == "pretty"

    def test_stdout_formatter_is_json_when_json_format_true(
        self, tmp_path: Path
    ) -> None:
        s = LoggingSettings(level=LogLevel.INFO, json_format=True, path=None)
        cfg = _build_dict_config(s, None)
        assert cfg["handlers"]["stdout"]["formatter"] == "json"

    def test_file_formatter_is_always_json(
        self, fresh_settings: LoggingSettings
    ) -> None:
        """Architecture §12.1: file is always JSON, regardless of json_format."""
        cfg = _build_dict_config(fresh_settings, Path("/tmp/x.log"))
        assert cfg["handlers"]["file"]["formatter"] == "json"

    def test_root_logger_has_all_handlers(
        self, fresh_settings: LoggingSettings
    ) -> None:
        cfg = _build_dict_config(fresh_settings, Path("/tmp/x.log"))
        root = cfg["loggers"][""]
        assert "stdout" in root["handlers"]
        assert "file" in root["handlers"]

    def test_root_logger_propagates(
        self, fresh_settings: LoggingSettings
    ) -> None:
        """Root logger must propagate so third-party loggers work."""
        cfg = _build_dict_config(fresh_settings, None)
        assert cfg["loggers"][""]["propagate"] is True

    def test_chatty_libraries_quieted(self, fresh_settings: LoggingSettings) -> None:
        """httpx / httpcore / sentence_transformers at WARNING."""
        cfg = _build_dict_config(fresh_settings, None)
        for noisy in ("httpx", "httpcore", "sentence_transformers"):
            assert cfg["loggers"][noisy]["level"] == logging.WARNING


class TestConfigureLogging:
    """configure_logging applies the dictConfig + structlog.configure."""

    def test_configure_logging_creates_root_handlers(
        self, fresh_settings: LoggingSettings, reset_logging
    ) -> None:
        configure_logging(fresh_settings)
        root = logging.getLogger()
        # Exactly the configured handlers — no leakage from prior tests.
        handler_names = {type(h).__name__ for h in root.handlers}
        assert "StreamHandler" in handler_names

    def test_configure_logging_is_idempotent(
        self, fresh_settings: LoggingSettings, reset_logging
    ) -> None:
        """Calling configure_logging twice doesn't double the handlers."""
        configure_logging(fresh_settings)
        configure_logging(fresh_settings)
        root = logging.getLogger()
        # dictConfig replaces handlers each call, so we should have
        # exactly the configured set — not 2x or 3x. We count *all*
        # handler types explicitly because ``WatchedFileHandler`` is
        # a subclass of ``StreamHandler`` and would otherwise inflate
        # the count.
        handler_types = sorted(type(h).__name__ for h in root.handlers)
        assert handler_types == ["StreamHandler", "WatchedFileHandler"]

    def test_configure_logging_with_unwritable_path_raises_logging_error(
        self, tmp_path: Path, reset_logging
    ) -> None:
        """A read-only parent directory raises LoggingError, not OSError."""
        # /proc/1/cmdline is a read-only pseudo-file on Linux; writing
        # to its parent is forbidden. We use the dev-null device as
        # the parent — it's not a directory, so mkdir will fail.
        s = LoggingSettings(
            level=LogLevel.INFO,
            json_format=False,
            path="/dev/null/cannot/log/here.log",
        )
        with pytest.raises(LoggingError):
            configure_logging(s)


class TestLogOutput:
    """End-to-end: log something, observe what comes out."""

    def test_pretty_stdout_includes_event_name(
        self, fresh_settings: LoggingSettings, reset_logging, capsys: pytest.CaptureFixture
    ) -> None:
        """Pretty stdout output contains the event name (not necessarily
        structured JSON)."""
        configure_logging(fresh_settings)
        log = get_logger("test.module")
        log.info("hello_world", key="value")
        captured = capsys.readouterr()
        # Pretty format puts the event name + key=value pairs on one
        # line. We don't assert exact format (structlog's ConsoleRenderer
        # is allowed to evolve) — just that the event name and key
        # appear.
        assert "hello_world" in captured.out
        assert "key" in captured.out
        assert "value" in captured.out

    def test_json_stdout_is_valid_json(
        self, tmp_path: Path, reset_logging, capsys: pytest.CaptureFixture
    ) -> None:
        """json_format=True → stdout gets parseable JSON per line."""
        s = LoggingSettings(level=LogLevel.INFO, json_format=True, path=None)
        configure_logging(s)
        log = get_logger("test.module")
        log.info("hello_world", count=42)
        captured = capsys.readouterr()
        # Each line is one JSON object.
        line = captured.out.strip().splitlines()[-1]
        record = json.loads(line)
        assert record["event"] == "hello_world"
        assert record["count"] == 42
        assert record["level"] == "info"
        assert record["logger"] == "test.module"
        assert "timestamp" in record

    def test_file_output_is_always_json(
        self, fresh_settings: LoggingSettings, reset_logging
    ) -> None:
        """Architecture §12.1: file is JSON regardless of json_format."""
        # Note: fresh_settings has json_format=False, so stdout will
        # be pretty. The file should still be JSON.
        configure_logging(fresh_settings)
        log = get_logger("test.module")
        log.warning("file_event_should_be_json", severity="high")
        # Force flush.
        for h in logging.getLogger().handlers:
            h.flush()
        # Read the file back.
        log_file = Path(fresh_settings.path)
        assert log_file.is_file(), f"log file was not created at {log_file}"
        contents = log_file.read_text(encoding="utf-8").strip()
        # Every non-empty line should be valid JSON.
        assert contents, "log file is empty"
        for line in contents.splitlines():
            record = json.loads(line)
            assert record["event"] == "file_event_should_be_json"
            assert record["severity"] == "high"
            assert record["level"] == "warning"
            assert "timestamp" in record
            assert "logger" in record

    def test_file_disabled_when_path_is_none(
        self, tmp_path: Path, reset_logging, capsys: pytest.CaptureFixture
    ) -> None:
        """path: None → no file handler, no file is created."""
        s = LoggingSettings(level=LogLevel.INFO, json_format=False, path=None)
        configure_logging(s)
        log = get_logger("test.module")
        log.info("event_without_file")
        # No file should be created anywhere under tmp_path.
        files_created = list(tmp_path.rglob("*.log"))
        assert files_created == []
        # The event still goes to stdout (captured by capsys).
        captured = capsys.readouterr()
        assert "event_without_file" in captured.out

    def test_file_dir_created_lazily(
        self, tmp_path: Path, reset_logging
    ) -> None:
        """A nested non-existent log-file parent directory is auto-created."""
        log_file = tmp_path / "deep" / "nested" / "subdir" / "test.log"
        assert not log_file.parent.exists()
        s = LoggingSettings(
            level=LogLevel.INFO, json_format=False, path=str(log_file)
        )
        configure_logging(s)
        # Parent dir was created.
        assert log_file.parent.is_dir()

    def test_existing_stdlib_logger_also_routed(
        self, fresh_settings: LoggingSettings, reset_logging, capsys: pytest.CaptureFixture
    ) -> None:
        """stdlib ``logging.getLogger`` calls also flow through our handlers.

        This matters because third-party libraries (urllib3, httpx,
        etc.) use stdlib logging, not structlog. The architecture
        doc §12.1 expects everything to land in the same file.
        """
        configure_logging(fresh_settings)
        stdlib_log = logging.getLogger("third_party.lib")
        stdlib_log.warning("stdlib_event_xyz")
        captured = capsys.readouterr()
        assert "stdlib_event_xyz" in captured.out


class TestLogLevels:
    """Level filtering works per the configured level."""

    def test_info_level_filters_debug(
        self, fresh_settings: LoggingSettings, reset_logging, capsys: pytest.CaptureFixture
    ) -> None:
        fresh_settings_dict = fresh_settings.model_copy(
            update={"level": LogLevel.INFO}
        )
        configure_logging(fresh_settings_dict)
        log = get_logger("test.module")
        log.debug("debug_event_should_be_filtered")
        log.info("info_event_should_pass")
        captured = capsys.readouterr()
        assert "debug_event_should_be_filtered" not in captured.out
        assert "info_event_should_pass" in captured.out

    def test_debug_level_passes_debug(
        self, fresh_settings: LoggingSettings, reset_logging, capsys: pytest.CaptureFixture
    ) -> None:
        s = fresh_settings.model_copy(update={"level": LogLevel.DEBUG})
        configure_logging(s)
        log = get_logger("test.module")
        log.debug("debug_event_now_passes")
        captured = capsys.readouterr()
        assert "debug_event_now_passes" in captured.out

    def test_error_level_filters_info(
        self, fresh_settings: LoggingSettings, reset_logging, capsys: pytest.CaptureFixture
    ) -> None:
        s = fresh_settings.model_copy(update={"level": LogLevel.ERROR})
        configure_logging(s)
        log = get_logger("test.module")
        log.info("info_event_filtered_at_error")
        log.error("error_event_passes")
        captured = capsys.readouterr()
        assert "info_event_filtered_at_error" not in captured.out
        assert "error_event_passes" in captured.out

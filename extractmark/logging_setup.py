"""Centralized logging setup for ExtractMark.

Creates one log file per benchmark run with a timestamp. All pipeline
output -- config, inference, evaluation, errors, and the Rich console
display -- is captured in the log file for post-run investigation.

Log files are saved to: logs/{run_name}_{timestamp}.log
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

from rich.console import Console

LOGS_DIR = Path("logs")

_log_file_path: Path | None = None
_logging_initialized: bool = False


def setup_logging(run_name: str, level: int = logging.INFO) -> Path:
    """Configure logging for a benchmark run.

    On the first call, creates a new log file and configures handlers.
    Subsequent calls reuse the same file (appending a section header),
    so the entire benchmark produces a single log file.

    Returns the path to the log file.
    """
    global _log_file_path, _logging_initialized

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not _logging_initialized:
        # First call: create the log file and set up handlers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{run_name}_{timestamp}.log"
        _log_file_path = LOGS_DIR / log_filename

        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(level)

        # File handler -- captures everything
        file_handler = logging.FileHandler(_log_file_path, mode="w")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

        # Console handler -- only warnings and above (Rich handles the pretty output)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(file_formatter)
        root.addHandler(console_handler)

        _logging_initialized = True

    # Write section header (first call = run header, subsequent calls = sub-run header)
    lgr = logging.getLogger("extractmark")
    lgr.info("=" * 60)
    lgr.info("ExtractMark Benchmark Run")
    lgr.info("Run name: %s", run_name)
    lgr.info("Log file: %s", _log_file_path)
    lgr.info("Started:  %s", datetime.now().isoformat())
    lgr.info("=" * 60)

    return _log_file_path


def reset_logging() -> None:
    """Reset logging state so the next setup_logging() creates a fresh file.

    Call this between independent benchmark invocations (e.g. in tests).
    """
    global _log_file_path, _logging_initialized
    _log_file_path = None
    _logging_initialized = False
    root = logging.getLogger()
    root.handlers.clear()


def get_log_file_path() -> Path | None:
    """Return the current log file path, if logging has been set up."""
    return _log_file_path


def create_rich_console_with_logging() -> Console:
    """Create a Rich Console that writes to both terminal and the log file.

    This ensures all Rich-formatted output (progress bars, tables, panels)
    is also captured in the log file as plain text.
    """
    if _log_file_path is None:
        return Console()

    log_file = open(_log_file_path, "a")
    file_console = Console(file=log_file, force_terminal=False, width=120, no_color=True)

    class TeeConsole(Console):
        """Console that writes to both terminal and a log file."""

        def __init__(self, file_console: Console, log_file):
            super().__init__()
            self._file_console = file_console
            self._log_file = log_file

        def print(self, *args, **kwargs):
            super().print(*args, **kwargs)
            # Also write to log file (stripped of color codes)
            try:
                self._file_console.print(*args, **kwargs)
                self._log_file.flush()
            except Exception:
                pass

        def rule(self, *args, **kwargs):
            super().rule(*args, **kwargs)
            try:
                self._file_console.rule(*args, **kwargs)
                self._log_file.flush()
            except Exception:
                pass

    return TeeConsole(file_console, log_file)

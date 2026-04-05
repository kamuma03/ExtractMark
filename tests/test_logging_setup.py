"""Tests for extractmark.logging_setup."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from extractmark.logging_setup import setup_logging, get_log_file_path, reset_logging, LOGS_DIR


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset logging between every test so each gets a fresh file."""
    reset_logging()
    yield
    reset_logging()


class TestSetupLogging:
    def test_creates_log_file(self, tmp_path):
        with patch("extractmark.logging_setup.LOGS_DIR", tmp_path):
            log_path = setup_logging("test_run")

        assert log_path.exists()
        assert "test_run" in log_path.name
        assert log_path.suffix == ".log"

    def test_log_file_has_header(self, tmp_path):
        with patch("extractmark.logging_setup.LOGS_DIR", tmp_path):
            log_path = setup_logging("header_test")

        content = log_path.read_text()
        assert "ExtractMark Benchmark Run" in content
        assert "header_test" in content

    def test_get_log_file_path(self, tmp_path):
        with patch("extractmark.logging_setup.LOGS_DIR", tmp_path):
            log_path = setup_logging("path_test")
            retrieved = get_log_file_path()

        assert retrieved == log_path

    def test_log_file_name_format(self, tmp_path):
        with patch("extractmark.logging_setup.LOGS_DIR", tmp_path):
            log_path = setup_logging("my_run")

        # Format: my_run_YYYYMMDD_HHMMSS.log
        assert log_path.name.startswith("my_run_")
        parts = log_path.stem.split("_")
        assert len(parts) >= 3  # name + date + time

    def test_second_call_reuses_same_file(self, tmp_path):
        """Calling setup_logging twice should produce ONE log file, not two."""
        with patch("extractmark.logging_setup.LOGS_DIR", tmp_path):
            path1 = setup_logging("run_a")
            path2 = setup_logging("run_b")

        assert path1 == path2  # same file
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1

        # Both headers should appear in the single file
        content = path1.read_text()
        assert "run_a" in content
        assert "run_b" in content

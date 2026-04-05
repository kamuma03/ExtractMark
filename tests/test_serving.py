"""Tests for serving infrastructure -- GPU monitor and vLLM server."""

from unittest.mock import MagicMock, patch
import subprocess

from extractmark.serving.gpu_monitor import get_gpu_memory_mb, get_gpu_info


class TestGPUMonitor:
    def test_get_gpu_memory_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "4096\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            mem = get_gpu_memory_mb()

        assert mem == 4096.0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "nvidia-smi" in args

    def test_get_gpu_memory_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            mem = get_gpu_memory_mb()
        assert mem is None

    def test_get_gpu_memory_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            mem = get_gpu_memory_mb()
        assert mem is None

    def test_get_gpu_memory_bad_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not-a-number\n"

        with patch("subprocess.run", return_value=mock_result):
            mem = get_gpu_memory_mb()
        assert mem is None

    def test_get_gpu_memory_nonzero_return(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            mem = get_gpu_memory_mb()
        assert mem is None

    def test_get_gpu_info_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GH200, 98304, 570.133.07\n"

        with patch("subprocess.run", return_value=mock_result):
            info = get_gpu_info()

        assert info is not None
        assert info["name"] == "NVIDIA GH200"
        assert info["memory_total_mb"] == 98304.0
        assert info["driver_version"] == "570.133.07"

    def test_get_gpu_info_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            info = get_gpu_info()
        assert info is None

    def test_get_gpu_info_bad_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "incomplete"

        with patch("subprocess.run", return_value=mock_result):
            info = get_gpu_info()
        assert info is None

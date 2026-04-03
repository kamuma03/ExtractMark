"""vLLM server lifecycle management.

Supports two modes:
- Docker: Uses nvcr.io/nvidia/vllm container (recommended for DGX Spark)
- Native: Uses locally installed vllm binary (requires compatible CUDA)
"""

from __future__ import annotations

import logging
import subprocess
import time

import openai

from extractmark.config import ModelConfig

logger = logging.getLogger(__name__)

VLLM_DOCKER_IMAGE = "nvcr.io/nvidia/vllm:25.09-py3"


class VLLMServer:
    """Manage a vLLM server process for a single model."""

    def __init__(self, model_id: str, config: ModelConfig, use_docker: bool = True):
        self.model_id = model_id
        self.config = config
        self.port = config.port
        self.use_docker = use_docker
        self.base_url = f"http://localhost:{self.port}/v1"
        self._process: subprocess.Popen | None = None
        self._container_name = f"extractmark-{model_id.lower()}"

    def _build_docker_command(self) -> list[str]:
        cmd = [
            "docker", "run",
            "--gpus", "all",
            "-p", f"{self.port}:8000",
            "--name", self._container_name,
            "--rm",
            "-v", f"{_hf_cache_dir()}:/root/.cache/huggingface",
            VLLM_DOCKER_IMAGE,
            "vllm", "serve", self.config.hf_model_id,
        ]
        cmd.extend(self.config.vllm_args)
        return cmd

    def _build_native_command(self) -> list[str]:
        cmd = [
            "vllm", "serve", self.config.hf_model_id,
            "--port", str(self.port),
        ]
        cmd.extend(self.config.vllm_args)
        return cmd

    def _build_command(self) -> list[str]:
        if self.use_docker:
            return self._build_docker_command()
        return self._build_native_command()

    def start(self, blocking: bool = False, timeout: int = 300) -> None:
        """Start the vLLM server.

        Args:
            blocking: If True, wait for the process to complete (foreground).
            timeout: Max seconds to wait for server readiness.
        """
        # Stop any existing container with the same name
        if self.use_docker:
            subprocess.run(
                ["docker", "rm", "-f", self._container_name],
                capture_output=True,
            )

        cmd = self._build_command()
        logger.info("Starting vLLM: %s", " ".join(cmd))

        if blocking:
            subprocess.run(cmd, check=True)
            return

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to become ready
        client = openai.OpenAI(base_url=self.base_url, api_key="not-needed")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                client.models.list()
                logger.info("vLLM server for %s is ready on port %d", self.model_id, self.port)
                return
            except Exception:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                    raise RuntimeError(
                        f"vLLM server for {self.model_id} exited unexpectedly: {stderr}"
                    )
                time.sleep(2)

        raise TimeoutError(
            f"vLLM server for {self.model_id} did not become ready within {timeout}s"
        )

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self.use_docker:
            subprocess.run(
                ["docker", "stop", self._container_name],
                capture_output=True,
            )
        if self._process and self._process.poll() is None:
            logger.info("Stopping vLLM server for %s", self.model_id)
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def is_running(self) -> bool:
        """Check if the server is alive."""
        if self.use_docker:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self._container_name],
                capture_output=True, text=True,
            )
            return result.stdout.strip() == "true"
        if self._process is None:
            return False
        return self._process.poll() is None

    def health_check(self) -> bool:
        """Check if the server is responding to API calls."""
        try:
            client = openai.OpenAI(base_url=self.base_url, api_key="not-needed")
            client.models.list()
            return True
        except Exception:
            return False


def _hf_cache_dir() -> str:
    """Get the HuggingFace cache directory."""
    import os
    return os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))

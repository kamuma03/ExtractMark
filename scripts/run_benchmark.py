#!/usr/bin/env python3
"""ExtractMark Full Benchmark Runner.

Manages the complete benchmark lifecycle:
1. Starts vLLM server for each model (native or Docker)
2. Runs benchmark pipeline with live progress, ETA, and status
3. Stops server between models to free GPU memory
4. Generates final report with findings

Usage:
    # Full benchmark (all models, all datasets, all metrics)
    python scripts/run_benchmark.py

    # Quick smoke test
    python scripts/run_benchmark.py --config configs/runs/quick_smoke.yaml

    # Specific models and datasets
    python scripts/run_benchmark.py -m M-01 -m M-08 -d D-01 -e L1 -e L2

    # Use Docker for vLLM serving
    python scripts/run_benchmark.py --docker

    # Skip server management (if vLLM is already running)
    python scripts/run_benchmark.py --no-serve
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from extractmark.config import load_config, ExtractMarkConfig, ModelConfig
from extractmark.pipeline import BenchmarkPipeline

console = Console()
# Logging is set up per-run inside the pipeline (see extractmark.logging_setup).
# The runner logs to the same file via the standard logging module.
logger = logging.getLogger("extractmark.runner")

# Models that need --trust-remote-code for vLLM
TRUST_REMOTE_CODE_MODELS = {"M-01", "M-06", "M-09"}

# Models that need extra pip dependencies inside the vLLM env
MODEL_EXTRA_DEPS = {
    "M-01": ["timm", "open_clip_torch", "albumentations"],
}

# vLLM GPU memory utilization (DGX Spark UMA needs headroom)
GPU_MEM_UTIL = 0.85


class VLLMServerManager:
    """Manage vLLM server lifecycle for benchmarking."""

    def __init__(self, use_docker: bool = False, port: int = 8000):
        self.use_docker = use_docker
        self.port = port
        self._process: subprocess.Popen | None = None
        self._container_name = "extractmark-vllm"

    def start(self, model_id: str, config: ModelConfig, timeout: int = 300) -> bool:
        """Start vLLM server for a model. Returns True if successful."""
        self.stop()  # Stop any existing server

        console.print(f"  Starting vLLM for [cyan]{model_id}[/cyan] ({config.hf_model_id})...")

        if self.use_docker:
            return self._start_docker(model_id, config, timeout)
        return self._start_native(model_id, config, timeout)

    def _start_native(self, model_id: str, config: ModelConfig, timeout: int) -> bool:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.hf_model_id,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        ]
        cmd.extend(config.vllm_args)
        if model_id in TRUST_REMOTE_CODE_MODELS:
            cmd.append("--trust-remote-code")

        logger.info("Starting vLLM: %s", " ".join(cmd))

        env = os.environ.copy()
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
        )

        return self._wait_ready(timeout)

    def _start_docker(self, model_id: str, config: ModelConfig, timeout: int) -> bool:
        # Stop existing container
        subprocess.run(["docker", "rm", "-f", self._container_name], capture_output=True)

        cmd = [
            "docker", "run", "-d",
            "--gpus", "all",
            "-p", f"{self.port}:8000",
            "--name", self._container_name,
            "--rm",
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
            "nvcr.io/nvidia/vllm:25.09-py3",
        ]

        # Install extra deps if needed
        extra_deps = MODEL_EXTRA_DEPS.get(model_id, [])
        if extra_deps:
            pip_cmd = f"pip install {' '.join(extra_deps)} && "
        else:
            pip_cmd = ""

        vllm_cmd = f"vllm serve {config.hf_model_id} --gpu-memory-utilization {GPU_MEM_UTIL}"
        for arg in config.vllm_args:
            vllm_cmd += f" {arg}"
        if model_id in TRUST_REMOTE_CODE_MODELS:
            vllm_cmd += " --trust-remote-code"

        cmd.extend(["bash", "-c", f"{pip_cmd}{vllm_cmd}"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Docker run failed: %s", result.stderr)
            return False

        return self._wait_ready(timeout)

    def _wait_ready(self, timeout: int) -> bool:
        """Poll health endpoint until ready."""
        import openai
        client = openai.OpenAI(
            base_url=f"http://localhost:{self.port}/v1", api_key="not-needed"
        )

        start = time.time()
        last_dot = start
        sys.stdout.write("  Waiting for server")
        sys.stdout.flush()

        while time.time() - start < timeout:
            try:
                client.models.list()
                elapsed = int(time.time() - start)
                console.print(f"\r  [green]Server ready[/green] ({elapsed}s)")
                return True
            except Exception:
                # Check if process died (native mode)
                if self._process and self._process.poll() is not None:
                    console.print("\r  [red]Server process died[/red]")
                    stdout = self._process.stdout.read().decode() if self._process.stdout else ""
                    logger.error("vLLM output: %s", stdout[-500:])
                    return False
                # Check if container died (docker mode)
                if self.use_docker:
                    check = subprocess.run(
                        ["docker", "ps", "-q", "-f", f"name={self._container_name}"],
                        capture_output=True, text=True,
                    )
                    if not check.stdout.strip():
                        console.print("\r  [red]Container died[/red]")
                        logs = subprocess.run(
                            ["docker", "logs", self._container_name],
                            capture_output=True, text=True,
                        )
                        logger.error("Container logs: %s", logs.stdout[-500:])
                        return False

                now = time.time()
                if now - last_dot > 5:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    last_dot = now
                time.sleep(2)

        console.print(f"\r  [red]Timeout after {timeout}s[/red]")
        return False

    def stop(self) -> None:
        """Stop the running vLLM server."""
        if self.use_docker:
            subprocess.run(
                ["docker", "rm", "-f", self._container_name],
                capture_output=True,
            )
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def health_check(self) -> bool:
        try:
            import openai
            client = openai.OpenAI(
                base_url=f"http://localhost:{self.port}/v1", api_key="not-needed"
            )
            client.models.list()
            return True
        except Exception:
            return False


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark pipeline."""
    config_path = Path(args.config)
    cfg = load_config(config_path)

    # Apply CLI overrides
    if args.models:
        cfg.run.models = args.models
    if args.datasets:
        cfg.run.datasets = args.datasets
    if args.evaluators:
        cfg.run.evaluators = args.evaluators
    if args.max_pages is not None:
        cfg.run.max_pages = args.max_pages

    if args.no_serve:
        # Run pipeline directly (assume server is already running)
        pipeline = BenchmarkPipeline(cfg)
        pipeline.run()
        return

    # Managed mode: start/stop vLLM per model, then run libraries (no server needed)
    server = VLLMServerManager(use_docker=args.docker, port=args.port)
    total_models = len(cfg.run.models)
    total_libraries = len(cfg.run.libraries)
    total_steps = total_models + (1 if total_libraries else 0)
    benchmark_start = time.time()

    console.print(Panel(
        f"[bold]ExtractMark Managed Benchmark[/bold]\n"
        f"Config: {config_path}\n"
        f"Models: {cfg.run.models}\n"
        f"Libraries: {cfg.run.libraries}\n"
        f"Datasets: {cfg.run.datasets}\n"
        f"Evaluators: {cfg.run.evaluators}\n"
        f"Serving: {'Docker' if args.docker else 'Native vLLM'}\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title="[cyan]ExtractMark[/cyan]",
    ))

    step = 0
    try:
        # Phase 1: Run vLLM-based models (one at a time, with server lifecycle)
        for i, model_id in enumerate(cfg.run.models):
            step += 1
            console.print()
            console.rule(
                f"[bold cyan]Model {i+1}/{total_models}: {model_id}[/bold cyan] "
                f"(step {step}/{total_steps})"
            )

            model_config = cfg.models.get(model_id)
            if not model_config:
                console.print(f"  [red]Model {model_id} not found in config, skipping[/red]")
                continue

            # Start server
            if not server.start(model_id, model_config):
                console.print(f"  [red]Failed to start server for {model_id}, skipping[/red]")
                continue

            # Run pipeline for this model only (no libraries in this pass)
            single_cfg = cfg.model_copy(deep=True)
            single_cfg.run.models = [model_id]
            single_cfg.run.libraries = []

            pipeline = BenchmarkPipeline(single_cfg)
            pipeline.run()

            # Stop server to free GPU memory
            console.print(f"  Stopping server for {model_id}...")
            server.stop()
            time.sleep(3)  # Brief pause for GPU memory cleanup

        # Phase 2: Run library adapters (no vLLM server needed)
        if cfg.run.libraries:
            step += 1
            console.print()
            console.rule(
                f"[bold magenta]Libraries: {', '.join(cfg.run.libraries)}[/bold magenta] "
                f"(step {step}/{total_steps})"
            )

            lib_cfg = cfg.model_copy(deep=True)
            lib_cfg.run.models = []  # No vLLM models in this pass

            pipeline = BenchmarkPipeline(lib_cfg)
            pipeline.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Cleaning up...[/yellow]")
    finally:
        server.stop()

    # Generate final consolidated report across all runs
    from extractmark.reporting.summary import SummaryReporter
    reporter = SummaryReporter(Path("results"), Path("report"))
    reporter.generate()

    total_elapsed = time.time() - benchmark_start
    console.print()
    console.print(Panel(
        f"[bold green]Benchmark Complete[/bold green]\n"
        f"Total time: {timedelta(seconds=int(total_elapsed))}\n"
        f"Results: results/\n"
        f"Report: report/benchmark_summary.md",
        title="Done",
    ))


def main():
    parser = argparse.ArgumentParser(
        description="ExtractMark Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py                                    # Full benchmark
  python scripts/run_benchmark.py --config configs/runs/quick_smoke.yaml
  python scripts/run_benchmark.py -m M-01 -m M-08 -d D-01 -e L1    # Specific
  python scripts/run_benchmark.py --no-serve                         # Server already running
  python scripts/run_benchmark.py --docker                           # Use Docker
        """,
    )
    parser.add_argument(
        "--config", "-c", default="configs/runs/full_benchmark.yaml",
        help="Run config YAML path",
    )
    parser.add_argument("--model", "-m", dest="models", action="append", help="Model IDs")
    parser.add_argument("--dataset", "-d", dest="datasets", action="append", help="Dataset IDs")
    parser.add_argument("--eval", "-e", dest="evaluators", action="append", help="Evaluator IDs")
    parser.add_argument("--max-pages", type=int, help="Max pages per dataset")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port (default: 8000)")
    parser.add_argument("--docker", action="store_true", help="Use Docker for vLLM")
    parser.add_argument("--no-serve", action="store_true", help="Skip server management")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()

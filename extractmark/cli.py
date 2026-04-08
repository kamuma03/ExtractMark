"""CLI entry point for ExtractMark."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from extractmark.config import load_config

app = typer.Typer(name="extractmark", help="ExtractMark benchmark pipeline")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to run config YAML"),
    models: list[str] | None = typer.Option(None, "--model", "-m", help="Override model IDs"),
    datasets: list[str] | None = typer.Option(None, "--dataset", "-d", help="Override dataset IDs"),
    evaluators: list[str] | None = typer.Option(None, "--eval", "-e", help="Override evaluator IDs"),
    libraries: list[str] | None = typer.Option(None, "--library", "-l", help="Override library IDs"),
    max_pages: int | None = typer.Option(None, "--max-pages", help="Limit pages per dataset"),
) -> None:
    """Run a benchmark pipeline from a config file."""
    from extractmark.pipeline import BenchmarkPipeline

    cfg = load_config(config)

    # Apply CLI overrides
    if models:
        cfg.run.models = models
    if datasets:
        cfg.run.datasets = datasets
    if evaluators:
        cfg.run.evaluators = evaluators
    if libraries:
        cfg.run.libraries = libraries
    if max_pages is not None:
        cfg.run.max_pages = max_pages

    console.print(f"[bold]ExtractMark[/bold] -- run: {cfg.run.name}")
    console.print(f"  Models:     {cfg.run.models}")
    console.print(f"  Libraries:  {cfg.run.libraries}")
    console.print(f"  Datasets:   {cfg.run.datasets}")
    console.print(f"  Evaluators: {cfg.run.evaluators}")
    console.print()

    pipeline = BenchmarkPipeline(cfg)
    pipeline.run()


@app.command()
def serve(
    model_id: str = typer.Argument(help="Model ID to serve (e.g. M-01)"),
    config: Path = typer.Option(
        Path("configs/models.yaml"), "--config", "-c", help="Path to models config"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port for vLLM server"),
    native: bool = typer.Option(False, "--native", help="Use native vllm binary instead of Docker"),
) -> None:
    """Start a vLLM server for a specific model (Docker by default)."""
    from extractmark.serving.vllm_server import VLLMServer

    cfg = load_config(config)
    model_config = cfg.models.get(model_id)
    if not model_config:
        console.print(f"[red]Model {model_id} not found in config[/red]")
        raise typer.Exit(1)

    model_config.port = port
    server = VLLMServer(model_id, model_config, use_docker=not native)
    mode = "native" if native else "Docker"
    console.print(f"[bold]Starting vLLM server for {model_id}[/bold] ({model_config.hf_model_id})")
    console.print(f"  Mode: {mode} | Port: {port}")
    server.start(blocking=True)


@app.command()
def report(
    results_dir: Path = typer.Argument(None, help="Run directory (default: latest run under results/)"),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Output directory (default: <results_dir>/report)"),
) -> None:
    """Generate benchmark summary report from existing results."""
    from extractmark.reporting.summary import SummaryReporter

    if results_dir is None:
        # Find the latest run folder under results/
        base = Path("results")
        if not base.exists():
            console.print("[red]No results/ directory found[/red]")
            raise typer.Exit(1)
        run_dirs = sorted(
            [d for d in base.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            console.print("[red]No run folders found under results/[/red]")
            raise typer.Exit(1)
        results_dir = run_dirs[0]
        console.print(f"Using latest run: [cyan]{results_dir}[/cyan]")

    if output_dir is None:
        output_dir = results_dir / "report"

    reporter = SummaryReporter(results_dir, output_dir)
    reporter.generate()
    console.print(f"[green]Report generated in {output_dir}[/green]")


if __name__ == "__main__":
    app()

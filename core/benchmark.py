"""Baseline comparison runner and markdown table renderer."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    name: str
    relative_l2: float
    inference_time_ms: float
    n_train_samples: int
    notes: str = ""


def render_markdown_table(results: list[BenchmarkResult]) -> str:
    """Render a list of BenchmarkResults as a GitHub-flavored markdown table."""
    header = "| Model | Rel L2 | Inference (ms) | Train samples | Notes |"
    sep    = "|-------|--------|----------------|---------------|-------|"
    rows = [
        f"| {r.name} | {r.relative_l2:.4f} | {r.inference_time_ms:.2f} | "
        f"{r.n_train_samples} | {r.notes} |"
        for r in results
    ]
    return "\n".join([header, sep] + rows)

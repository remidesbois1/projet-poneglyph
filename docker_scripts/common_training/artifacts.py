from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def standard_pipeline_summary(
    *,
    status: str,
    training_kind: str,
    provider: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    final_model_dir: str | Path,
    hf_repo: str,
    benchmark_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "training_kind": training_kind,
        "provider": provider,
        "training_job_id": os.getenv("TRAINING_JOB_ID") or os.getenv("PONEGLYPH_TRAINING_JOB_ID"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "final_model_dir": str(final_model_dir),
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
        "comparison_path": str(comparison_path) if comparison_path else None,
        "hf_repo": hf_repo,
        "finished_at": utc_now_iso() if status in {"complete", "completed", "failed"} else None,
    }
    if error_message:
        summary["error_message"] = error_message
    return summary

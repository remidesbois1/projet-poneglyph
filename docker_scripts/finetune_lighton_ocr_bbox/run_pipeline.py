import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login


os.environ["PYTHONUNBUFFERED"] = "1"
# The classic HTTP/LFS uploader proved more reliable on the user's Windows/WSL
# setup than the Xet path for multi-GB safetensors.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
os.chdir(SCRIPT_DIR)
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_HF_REPO = "Remidesbois/LightonOCR-2-1b-poneglyph-bbox"


def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LightOnOCR bbox fine-tune pipeline.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-remote", action="store_true")
    return parser.parse_args()


def dataset_dir():
    return Path(os.getenv("LIGHTON_BBOX_DATASET_DIR", SCRIPT_DIR / "lighton_bbox_dataset"))


def output_dir():
    return Path(os.getenv("LIGHTON_BBOX_OUTPUT_DIR", SCRIPT_DIR / "outputs_lighton_bbox"))


def final_model_dir():
    return output_dir() / "final_merged"


def hf_repo_id():
    return os.getenv("HF_REPO", DEFAULT_HF_REPO)


def run_step(label, script, *args):
    print(f"\n{label}", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", script, *args],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if result.returncode:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def dataset_readiness(path):
    for split in ("train", "val", "test"):
        metadata = path / split / "metadata.jsonl"
        if not metadata.is_file() or metadata.stat().st_size <= 0:
            return False, f"missing/empty {split}/metadata.jsonl"
        try:
            first = next(
                (json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()),
                None,
            )
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid {split}/metadata.jsonl ({exc})"
        if not isinstance(first, dict):
            return False, f"no samples in {split}/metadata.jsonl"
        image_file = str(first.get("image_file") or "").strip()
        if not image_file or not (path / split / image_file).is_file():
            return False, f"missing image referenced by first {split} sample"
    return True, "ready"


def missing_required_env(*, require_export=False):
    missing = []
    if require_export:
        missing.extend(
            name
            for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
            if not os.getenv(name)
        )
    if env_bool("LIGHTON_BBOX_REQUIRE_UPLOAD", False) and not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN")
    return missing


def check_hf_access():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing; remote HF check skipped.", flush=True)
        return
    info = HfApi(token=token).whoami()
    print(f"Hugging Face token OK for {info.get('name') or 'authenticated user'}.", flush=True)


def dry_run(check_remote=False):
    failures = []
    for path in (dataset_dir(), output_dir()):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            failures.append(f"not writable: {path} ({exc})")
    ready, _reason = dataset_readiness(dataset_dir())
    missing = missing_required_env(
        require_export=not ready or env_bool("LIGHTON_BBOX_FORCE_EXPORT", False)
    )
    if missing:
        failures.append(f"missing env: {', '.join(missing)}")
    try:
        run_step("Checking LightOn bbox runtime and prompt contract", "smoke_check.py")
    except Exception as exc:
        failures.append(str(exc))
    if check_remote:
        try:
            check_hf_access()
        except Exception as exc:
            failures.append(f"Hugging Face check failed: {exc}")
    if failures:
        print("Dry run failed:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    print("LightOn bbox dry run passed.", flush=True)
    return 0


def maybe_upload_to_hf():
    if env_bool("LIGHTON_BBOX_SKIP_UPLOAD", False):
        print("Hugging Face upload skipped by LIGHTON_BBOX_SKIP_UPLOAD=1.", flush=True)
        return "skipped", None
    token = os.getenv("HF_TOKEN")
    if not token:
        message = "HF_TOKEN missing; optional Hugging Face upload skipped."
        if env_bool("LIGHTON_BBOX_REQUIRE_UPLOAD", False):
            raise RuntimeError(message)
        print(message, flush=True)
        return "skipped", message
    try:
        login(token=token)
        api = HfApi(token=token)
        repo = hf_repo_id()
        api.create_repo(repo_id=repo, exist_ok=True, private=env_bool("HF_PRIVATE", False))
        print(f"Uploading final LightOn bbox model to {repo}...", flush=True)
        api.upload_folder(
            folder_path=str(final_model_dir()),
            repo_id=repo,
            repo_type="model",
            commit_message="Upload LightOnOCR Poneglyph bbox fine-tuned model",
        )
        print("Hugging Face upload complete.", flush=True)
        return "complete", None
    except Exception as exc:
        message = str(exc)
        if env_bool("LIGHTON_BBOX_REQUIRE_UPLOAD", False):
            raise
        print(
            "WARNING: training and benchmark are complete, but Hugging Face upload failed: "
            f"{message}",
            flush=True,
        )
        return "failed", message


def write_summary(status, error=None, *, upload_status=None, upload_error=None):
    output_dir().mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "error_message": error,
        "training_kind": "lighton_bbox",
        "dataset_dir": str(dataset_dir()),
        "output_dir": str(output_dir()),
        "final_model_dir": str(final_model_dir()),
        "benchmark_path": str(final_model_dir() / "benchmark_lighton_bbox.json"),
        "hardware_profile_path": str(output_dir() / "hardware_profile.json"),
        "hf_repo": hf_repo_id(),
        "upload_status": upload_status,
        "upload_error": upload_error,
    }
    (output_dir() / "pipeline_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main():
    args = parse_args()
    if args.dry_run or env_bool("LIGHTON_BBOX_DRY_RUN", False):
        raise SystemExit(dry_run(args.check_remote))

    try:
        print("Starting LightOnOCR Poneglyph bbox pipeline.", flush=True)
        print(f"Dataset: {dataset_dir()}", flush=True)
        print(f"Output:  {output_dir()}", flush=True)
        print(f"HF repo: {hf_repo_id()}", flush=True)

        ready, reason = dataset_readiness(dataset_dir())
        force_export = env_bool("LIGHTON_BBOX_FORCE_EXPORT", False)
        needs_export = force_export or not ready
        missing = missing_required_env(require_export=needs_export)
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
        if ready and not force_export:
            print("Frozen dataset already exists; export skipped.", flush=True)
        else:
            if dataset_dir().exists() and not ready:
                print(f"Dataset incomplete ({reason}); re-exporting.", flush=True)
            run_step("Step 1: exporting frozen full-page bbox dataset", "export_dataset.py")

        model_ready = (final_model_dir() / "config.json").is_file()
        benchmark_ready = (final_model_dir() / "benchmark_lighton_bbox.json").is_file()
        force_train = env_bool("LIGHTON_BBOX_FORCE_TRAIN", False)
        if model_ready and benchmark_ready and not force_train:
            print("Final model and benchmark already exist; training skipped.", flush=True)
        elif model_ready and not force_train:
            run_step(
                "Step 2: benchmarking existing final model",
                "train_lighton_bbox.py",
                "--benchmark-only",
            )
        else:
            resume_args = ()
            if any(output_dir().glob("checkpoint-*")):
                print("Training checkpoint found; resuming automatically.", flush=True)
                resume_args = ("--resume", "auto")
            run_step(
                "Step 2: RTX 5090 optimized LightOn bbox fine-tuning + benchmark",
                "train_lighton_bbox.py",
                *resume_args,
            )

        upload_status, upload_error = maybe_upload_to_hf()
        summary = write_summary(
            "complete",
            upload_status=upload_status,
            upload_error=upload_error,
        )
        print("LightOn bbox pipeline complete.", flush=True)
        return summary
    except Exception as exc:
        write_summary("failed", str(exc))
        print(f"LightOn bbox pipeline failed: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()

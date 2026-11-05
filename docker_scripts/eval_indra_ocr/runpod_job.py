"""RunPod pod for the Indra recognizer fine-tune: create, ssh/scp, guard, delete.

State lives in outputs/runpod-state.json and every command refuses a pod this
script did not create.  Credentials come from runpodctl's saved config and the
SSH key registered for Poneglyph training; neither enters the uploaded archive.

  python runpod_job.py create --gpu "NVIDIA GeForce RTX 4090" --hours 1.5
  python runpod_job.py upload outputs/indra_rec_training.tar /workspace/indra/
  python runpod_job.py exec "cd /workspace/indra && tar xf indra_rec_training.tar"
  python runpod_job.py download /workspace/indra/runs.tar outputs/
  python runpod_job.py guard        # stops the pod at the deadline (run in background)
  python runpod_job.py delete
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrain_ppocr_pipeline"))
from runpod_cli import run  # noqa: E402  (runpodctl with the saved API key)

TAG = "poneglyph-indra-rec-20261105"
STATE = Path(__file__).resolve().parent / "outputs/runpod-state.json"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
KEY = Path.home() / ".runpod/ssh/poneglyph-training-20261103"
KNOWN = Path.home() / ".runpod/ssh/poneglyph-indra-20261105-known-hosts"


def ctl(*args) -> dict | list:
    res = run(*args, "-o", "json", capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"runpodctl {' '.join(args)} failed: {res.stderr or res.stdout}")
    return json.loads(res.stdout) if res.stdout.strip() else {}


def state() -> dict:
    s = json.loads(STATE.read_text(encoding="utf-8"))
    if s.get("created_by") != TAG or not s.get("pod_id"):
        raise ValueError("refusing to act on a pod this job did not create")
    return s


def ssh_target() -> tuple[list[str], str, str]:
    s = state()
    info = s.get("ssh")
    if not info:
        pod = ctl("pod", "get", s["pod_id"])
        info = (pod.get("ssh") or {})
        if not info.get("ip") and pod.get("publicIp"):
            port = next((p["publicPort"] for p in pod.get("portMappings", []) if p.get("privatePort") == 22), None)
            info = {"ip": pod["publicIp"], "port": port}
        if not info.get("ip") or not info.get("port"):
            raise RuntimeError(f"no public SSH yet: {json.dumps(pod)[:400]}")
        s["ssh"] = {"ip": info["ip"], "port": info["port"]}
        STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")
        info = s["ssh"]
    common = ["-i", str(KEY), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
              "-o", f"UserKnownHostsFile={KNOWN}", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
    return common, str(info["port"]), f"root@{info['ip']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["create", "exec", "upload", "download", "guard", "delete", "status"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--startup-minutes", type=float, default=10)
    a = ap.parse_args()
    if a.action == "create":
        if STATE.exists() and not json.loads(STATE.read_text()).get("deleted_utc"):
            raise SystemExit("a pod from this job is still recorded; delete it first")
        # No --wait: a timed-out wait keeps the pod alive and billing.  Record the pod first,
        # then wait here and delete it ourselves if it never becomes reachable.
        pod = ctl("pod", "create", "--image", IMAGE, "--gpu-id", a.gpu, "--name", TAG, "--container-disk-in-gb", "40",
                  "--ports", "22/tcp", "--cloud-type", "SECURE")
        now = datetime.now(timezone.utc)
        STATE.write_text(json.dumps({"created_by": TAG, "pod_id": pod["id"], "gpu": a.gpu, "image": IMAGE,
                                     "created_utc": now.isoformat(), "deadline_utc": (now + timedelta(hours=a.hours)).isoformat(),
                                     "cost_per_hr": pod.get("costPerHr")}, indent=2), encoding="utf-8")
        print("created", pod["id"], pod.get("costPerHr"), flush=True)
        limit = time.time() + a.startup_minutes * 60
        while time.time() < limit:
            try:
                common, port, host = ssh_target()
                if subprocess.call(["ssh", *common, "-p", port, host, "nvidia-smi -L"]) == 0:
                    print("ready", host, port)
                    return
            except RuntimeError:
                pass
            time.sleep(20)
        res = run("pod", "delete", pod["id"], capture_output=True, text=True)
        s = json.loads(STATE.read_text(encoding="utf-8"))
        s["deleted_utc"] = datetime.now(timezone.utc).isoformat()
        s["deleted_reason"] = f"not reachable after {a.startup_minutes} min"
        STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")
        raise SystemExit(f"pod never became reachable; deleted (rc={res.returncode})")
    elif a.action == "status":
        print(json.dumps(ctl("pod", "get", state()["pod_id"]), indent=1)[:3000])
    elif a.action == "exec":
        common, port, host = ssh_target()
        raise SystemExit(subprocess.call(["ssh", *common, "-p", port, host, a.args[0]]))
    elif a.action == "upload":
        src, dst = a.args
        if not dst.startswith("/workspace/"):
            raise ValueError("uploads stay inside /workspace")
        common, port, host = ssh_target()
        raise SystemExit(subprocess.call(["scp", *common, "-P", port, str(Path(src).resolve()), f"{host}:{dst}"]))
    elif a.action == "download":
        src, dst = a.args
        if not src.startswith("/workspace/"):
            raise ValueError("only task workspace artifacts")
        common, port, host = ssh_target()
        raise SystemExit(subprocess.call(["scp", *common, "-P", port, f"{host}:{src}", str(Path(dst).resolve())]))
    elif a.action == "guard":
        s = state()
        deadline = datetime.fromisoformat(s["deadline_utc"]).timestamp()
        while time.time() < deadline:
            if json.loads(STATE.read_text()).get("deleted_utc"):
                return
            time.sleep(30)
        res = run("pod", "stop", s["pod_id"], capture_output=True, text=True)
        print("deadline reached, pod stop:", res.returncode, res.stdout, res.stderr)
    elif a.action == "delete":
        s = state()
        res = run("pod", "delete", s["pod_id"], capture_output=True, text=True)
        if res.returncode:
            raise SystemExit(f"delete failed: {res.stderr or res.stdout}")
        s["deleted_utc"] = datetime.now(timezone.utc).isoformat()
        STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print("deleted", s["pod_id"])


if __name__ == "__main__":
    main()

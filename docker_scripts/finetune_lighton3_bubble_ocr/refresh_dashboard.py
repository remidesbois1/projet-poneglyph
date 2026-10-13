"""Refresh final local links, scoring provenance and publication receipt."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune_surya_bubble_ocr"))
from bubble_monitor import write_dashboard
from training_monitor import atomic_json


def refresh(directory):
    summary_path=directory/"training_summary.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    config=json.loads((directory/"run_config.json").read_text(encoding="utf-8"))
    summary["training_curve_scoring_version"]=config.get("training_curve_scoring_version")
    summary["scoring_version"]=config.get("scoring_version")
    receipt=directory/"hf_publication.json"
    if receipt.exists():
        summary["hf_publication"]=json.loads(receipt.read_text(encoding="utf-8"))
        summary["model_url"]="https://huggingface.co/"+summary["hf_publication"]["repo_id"]
    history=[json.loads(line) for line in (directory/"metrics.jsonl").read_text(encoding="utf-8").splitlines()
             if json.loads(line).get("event")=="metrics"]
    atomic_json(summary_path,summary)
    write_dashboard(directory/"training_dashboard.html",summary,history)
    print(json.dumps({"status":summary["status"],"test_cer":summary["test_cer"],"model_url":summary.get("model_url")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    refresh(parser.parse_args().directory)

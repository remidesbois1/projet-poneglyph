"""Fresh validated page export, using Poneglyph's existing private-R2 handling."""

import sys
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finetune_surya_ocr_bbox import export_dataset as source_export
from prepare_dataset import prepare


def audit_annotations(source):
    """Rejected OCR candidates are negatives; pending/unchecked zones are not."""
    from supabase import create_client
    db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    page_ids = [json.loads(line)["page_id"] for split in ("train", "val", "test")
                for line in (source / split / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for start in range(0, len(page_ids), 100):
        offset = 0
        while True:
            batch = db.table("bulles").select("id_page,statut,order").in_("id_page", page_ids[start:start+100]).order("id").range(offset, offset+999).execute().data
            rows.extend(batch)
            if len(batch) < 1000:
                break
            offset += 1000
    incomplete, rejected = defaultdict(list), set()
    for row in rows:
        if row["statut"] == "Rejeté":
            rejected.add(str(row["id_page"]))
        elif row["statut"] != "Validé" or row["order"] is None:
            incomplete[str(row["id_page"])].append({"status": row["statut"], "missing_order": row["order"] is None})
    report = {"dataset_pages": len(page_ids), "bubble_statuses": dict(Counter(row["statut"] for row in rows)),
              "incomplete_accepted_pages": len(incomplete), "pages_with_rejected_candidates": len(rejected), "incomplete_details": dict(incomplete)}
    (source / "annotation_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if incomplete:
        raise ValueError(f"{len(incomplete)} page(s) contain pending zones or missing reading order; finish page annotation before full-page SFT")
    print("Annotation completeness audit passed", flush=True)
    return report


if __name__ == "__main__":
    destination = Path("/workspace/dataset")
    source_export.OUTPUT_DIR = Path("/workspace/dataset_source")
    source_export.TARGET_LONGEST_SIDE = 2048
    source_export.REQUIRE_ORDER = True
    source_export.CLEAN_DATASET = False
    source_export.main()
    audit = audit_annotations(source_export.OUTPUT_DIR)
    prepare(source_export.OUTPUT_DIR, destination)
    (destination / "annotation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

#!/usr/bin/env python3
"""
Quick schema check for result JSONs.
Validates that bounding boxes use [x, y, h, w] and required keys exist.
Usage:
  python scripts/utilities/schema_check.py /path/to/results/<DATASET_ID>
"""

import sys
import json
from pathlib import Path

REQUIRED_ELEMENT_KEYS = {"type", "bbox"}

def is_xyhw(bbox):
    # Expect list of 4 ints/floats in [x, y, h, w]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x, y, h, w = bbox
    try:
        _ = float(x); _ = float(y); _ = float(h); _ = float(w)
    except Exception:
        return False
    return True

def check_file(fp: Path):
    try:
        data = json.loads(fp.read_text(encoding='utf-8'))
    except Exception as e:
        return [(str(fp), f"invalid_json: {e}")]
    issues = []
    elements = data.get("elements", [])
    if not isinstance(elements, list):
        issues.append((str(fp), "elements_not_list"))
        return issues
    for idx, el in enumerate(elements):
        if not isinstance(el, dict):
            issues.append((str(fp), f"element_{idx}_not_object"))
            continue
        missing = REQUIRED_ELEMENT_KEYS - set(el.keys())
        if missing:
            issues.append((str(fp), f"element_{idx}_missing_keys:{sorted(missing)}"))
        if "bbox" in el and not is_xyhw(el["bbox"]):
            issues.append((str(fp), f"element_{idx}_bbox_not_xyhw"))
    return issues

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/utilities/schema_check.py /path/to/results/<DATASET_ID>")
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Path not found: {root}")
        sys.exit(2)
    json_files = list(root.rglob("*.json"))
    total, bad = 0, 0
    for fp in json_files:
        total += 1
        issues = check_file(fp)
        if issues:
            bad += 1
            for f, msg in issues:
                print(f"[ISSUE] {f}: {msg}")
    print(f"Checked: {total} files; Issues: {bad}")
    sys.exit(0 if bad == 0 else 3)

if __name__ == "__main__":
    main()



import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
from typing import List, Dict

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.infer_page import PS05Pipeline

COLOR_ORDER = {
    "Background": (200, 200, 200),
    "Text": (0, 153, 255),
    "Title": (255, 165, 0),
    "List": (76, 175, 80),
    "Table": (156, 39, 176),
    "Figure": (244, 67, 54),
}


def collect_image_paths(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted([x for x in p.glob("**/*") if x.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def normalize_output_for_stage(results: Dict, stage: int) -> Dict:
    # Conform to challenge language and shapes.
    out = {
        "page": int(results.get("page", 1)),
        "size": results.get("size", {}),
        "elements": [],
        "preprocess": results.get("preprocess", {}),
    }
    # Stage 1 elements (layout)
    for el in results.get("elements", []):
        out["elements"].append({
            "id": el.get("id"),
            "cls": el.get("cls"),
            "bbox": el.get("bbox"),
            "score": float(el.get("score", 0.0)),
        })
    if stage >= 2:
        out["text_lines"] = []
        for tl in results.get("text_lines", []):
            out["text_lines"].append({
                "bbox": tl.get("bbox"),
                "text": tl.get("text", ""),
                "lang": tl.get("lang", "und"),
                "score": float(tl.get("score", 0.0)),
            })
    if stage >= 3:
        # Optional complex elements summaries
        for key in ("tables", "figures", "charts", "maps"):
            if key in results:
                out[key] = []
                for item in results.get(key, []):
                    norm = {
                        "bbox": item.get("bbox"),
                        "summary": item.get("summary", ""),
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                    if key == "charts":
                        norm["type"] = item.get("type", "unknown")
                    out[key].append(norm)
    return out


def write_json(output_dir: Path, image_path: Path, data: Dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def zip_dir(src_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                fp = Path(root) / fn
                z.write(fp, arcname=str(fp.relative_to(src_dir)))


def main():
    parser = argparse.ArgumentParser(description="PS-05 submission packager")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", required=True, help="Output directory for JSON + zip")
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3], help="Processing stage")
    parser.add_argument("--zip-name", default="submission.zip", help="Name of the zip file")
    parser.add_argument("--config", default="configs/ps05_config.yaml", help="Optional config yaml path")
    args = parser.parse_args()

    images = collect_image_paths(args.input)
    if not images:
        raise SystemExit("No images found to process")

    pipeline = PS05Pipeline(config_path=args.config)
    out_dir = Path(args.output)
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for img in images:
        try:
            res = pipeline.process_image(str(img), stage=args.stage)
            norm = normalize_output_for_stage(res, stage=args.stage)
            write_json(json_dir, img, norm)
            index.append({"image": img.name, "json": f"json/{img.stem}.json"})
        except Exception as e:
            print(f"Failed: {img} -> {e}")

    # Write an index file for convenience
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump({"files": index, "stage": args.stage}, f, indent=2)

    # Zip up
    zip_path = out_dir / args.zip_name
    zip_dir(json_dir, zip_path)
    print(f"Submission package created: {zip_path}")


if __name__ == "__main__":
    main() 
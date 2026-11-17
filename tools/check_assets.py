"""Check external weight artifacts without downloading or loading pickle files."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def check(manifest, root, stage="inference"):
    entries = {}
    if stage in {"inference", "all"}:
        entries.update({f"seed_{k}": v for k, v in manifest["seeds"].items()})
    if stage in {"training", "all"}:
        entries.update(manifest["training_artifacts"])
    rows = []
    for name, spec in entries.items():
        path = root / spec["file"]
        status = "missing"
        if path.is_file():
            if "bytes" in spec and path.stat().st_size != spec["bytes"]:
                status = "size_mismatch"
            else:
                status = "ok" if digest(path) == spec["sha256"] else "sha256_mismatch"
        rows.append({"artifact": name, "file": spec["file"], "status": status})
    return {"stage": stage, "complete": bool(rows) and all(r["status"] == "ok" for r in rows),
            "files": rows, "download_performed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "weight_manifest.json")
    p.add_argument("--weights-root", type=Path, default=Path("weights"))
    p.add_argument("--stage", choices=["inference", "training", "all"], default="inference")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    result = check(json.loads(args.manifest.read_text(encoding="utf8")), args.weights_root, args.stage)
    content = json.dumps(result, indent=2) + "\n"
    print(content, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content, encoding="utf8")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

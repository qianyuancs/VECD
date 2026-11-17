"""Populate a protocol's data tree from prepared manifests with matching record IDs."""
import argparse
from filecmp import cmp
import json
from pathlib import Path
from shutil import copyfile


def stage(protocol, sources, root):
    root = root.resolve()
    entries = {}
    for manifest in sources:
        for row in json.loads(manifest.read_text(encoding="utf8"))["records"]:
            if row["id"] in entries:
                raise ValueError(f"Duplicate source ID: {row['id']}")
            entries[row["id"]] = (manifest.parent, row)
    files = {}
    for ref in json.loads(protocol.read_text(encoding="utf8"))["records"]:
        folder, row = entries[ref["id"]]
        if set(row["positive"]) != set(ref["positive"]) or set(row["negative"]) != set(ref["negative"]):
            raise ValueError(f"Attribute labels differ: {ref['id']}")
        if row.get("rgb_sha") and ref.get("rgb_sha") and row["rgb_sha"] != ref["rgb_sha"]:
            raise ValueError(f"RGB identity differs: {ref['id']}")
        pairs = [(row[key], ref[key]) for key in ("before", "after", "mask", "valid", "static_labels") if key in ref]
        pairs += [(row["targets"][k], dest) for k, dest in ref.get("targets", {}).items()]
        for source, relative in pairs:
            path, dest = (folder/source).resolve(), (root/relative).resolve()
            if not dest.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing source or invalid destination: {path}, {relative}")
            if dest in files and files[dest] != path and not cmp(files[dest], path, shallow=False):
                raise ValueError(f"Conflicting data for {relative}")
            files[dest] = path
    for dest, source in files.items():
        if dest.exists():
            if not cmp(source, dest, shallow=False):
                raise ValueError(f"Existing file differs: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, dest)
    return len(files)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("protocol", type=Path, help="Target records.json with the desired asset paths")
    p.add_argument("sources", type=Path, nargs="+", help="Prepared records.json files")
    p.add_argument("--data-root", type=Path, default=Path("data/prepared"))
    args = p.parse_args()
    print(f"{stage(args.protocol, args.sources, args.data_root)} files staged or verified")

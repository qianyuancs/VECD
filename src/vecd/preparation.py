"""Convert image/annotation exports to portable records, without legacy imports."""
from pathlib import Path
import os

import numpy as np
from PIL import Image

from .io import read_json, resolve, write_json
from .model import FAMILIES

VALUES = (("blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"),
          ("large", "small"), ("cube", "cylinder", "sphere"), ("metal", "rubber"))


def source_reference(root, output, value):
    source = resolve(root, value).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        return os.path.relpath(source, Path(output).resolve())
    except ValueError:
        # Windows cannot form a relative path between different drives.
        return str(source)


def image_mask(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, np.uint8)*255).save(path)


def prepare_annotated_pairs(annotations, root, output):
    """Input per pair: masks_before/after NPZ, object attributes and edit list.

    The metadata file may describe single-edit or native multi-edit scenes.
    Different-family temporal overlaps are ignored, not assigned arbitrarily.
    """
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in read_json(annotations):
        ident = row["id"]
        before_objects = {o["id"]: o for o in row["objects_before"]}
        after_objects = {o["id"]: o for o in row["objects_after"]}
        if set(before_objects) != set(after_objects):
            raise ValueError("This four-attribute release does not label object addition/removal")
        actual = {(i, f) for i in before_objects for f in FAMILIES
                  if before_objects[i][f] != after_objects[i][f]}
        declared = {(e["object_id"], e["family"]) for e in row["edits"]}
        if actual != declared or len({e["object_id"] for e in row["edits"]}) != len(row["edits"]):
            raise ValueError("Exact rendered data require one declared changed attribute per edited object")
        from hashlib import sha256
        folder = Path("annotations")/sha256(ident.encode()).hexdigest()
        masks, ids = [], []
        labels = []
        for time, side in enumerate(("before", "after")):
            objects = row["objects_"+side]
            with np.load(resolve(root, row["masks_"+side]), allow_pickle=False) as z:
                mm = z["masks"].astype(bool)
                ii = z["object_ids"].tolist() if "object_ids" in z else [o["id"] for o in objects]
            if (mm.sum(0) > 1).any():
                raise ValueError("Visible instance masks must be disjoint in each frame")
            masks.append(mm)
            ids.append(ii)
            label = np.full((4, *mm.shape[1:]), 255, np.uint8)
            for obj in objects:
                for k, family in enumerate(FAMILIES):
                    label[k, mm[ii.index(obj["id"])]] = VALUES[k].index(obj[family])
            labels.append(label)
        targets = {f: np.zeros_like(masks[0][0]) for f in FAMILIES}
        for edit in row["edits"]:
            family = edit["family"]
            if family not in targets:
                raise ValueError("This release only queries the four attribute families")
            object_id = edit["object_id"]
            targets[family] |= masks[0][ids[0].index(object_id)] | masks[1][ids[1].index(object_id)]
        coverage = np.stack(list(targets.values())).sum(0)
        valid = coverage <= 1
        targets = {f: target & valid for f, target in targets.items()}
        change = np.stack(list(targets.values())).any(0)
        image_mask(output/folder/"change.png", change)
        image_mask(output/folder/"valid.png", valid)
        for family, target in targets.items():
            image_mask(output/folder/(family+".png"), target)
        label_path = output/folder/"static_labels.npz"
        np.savez_compressed(label_path, labels=np.stack(labels))
        clean = {k:row[k] for k in ("id", "group", "split", "before", "after")}
        for key in ("before", "after"):
            clean[key] = source_reference(root, output, row[key])
        clean.update(mask=str(folder/"change.png"), valid=str(folder/"valid.png"),
                     targets={f:str(folder/(f+".png")) for f in FAMILIES},
                     static_labels=str(folder/"static_labels.npz"),
                     positive=[f for f,m in targets.items() if (m & valid).any()],
                     negative=[f for f,m in targets.items() if not (m & valid).any()])
        if "rgb_sha" in row:
            clean["rgb_sha"] = row["rgb_sha"]
        rows.append(clean)
    manifest = dict(version=1, records=rows, masks="exact visible-instance temporal unions")
    write_json(output/"records.json", manifest)
    return manifest


def prepare_natural(metadata, root, output):
    """Normalize white=edit masks; labels must already be declared in metadata.

    MagicBrush original RGBA mask uses transparent pixels for the edited ROI.
    No instruction parser silently certifies the absence of an attribute.
    """
    root, output = Path(root), Path(output)
    rows = []
    for row in read_json(metadata):
        clean = {k:row[k] for k in ("id", "group", "split", "before", "after", "positive", "negative")}
        for key in ("before", "after"):
            clean[key] = source_reference(root, output, row[key])
        if set(clean["positive"]) & set(clean["negative"]):
            raise ValueError("An attribute cannot be both positive and negative")
        with Image.open(resolve(root, row["edit_mask"])) as image:
            if row.get("mask_encoding") == "transparent_is_edit":
                if "A" not in image.getbands():
                    raise ValueError("Expected an alpha channel")
                mask = np.asarray(image.getchannel("A")) < 128
            elif row.get("mask_encoding", "white_is_edit") == "white_is_edit":
                mask = np.asarray(image.convert("L")) > 127
            else:
                raise ValueError("Unknown mask convention")
        from hashlib import sha256
        path = Path("annotations")/(sha256(row["id"].encode()).hexdigest()+".png")
        image_mask(output/path, mask)
        clean.update(mask=str(path), annotation_quality="edit ROI proxy; not certified per-attribute GT")
        rows.append(clean)
    manifest = dict(version=1, records=rows, masks="caption-conditioned edit ROI proxies")
    write_json(output/"records.json", manifest)
    return manifest

"""Manifest-based data; masks enter forward only when explicitly requested."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from .features import feature_path
from .io import load_mask, read_json, resize_mask, resolve
from .model import FAMILIES


class Records:
    def __init__(self, manifest, root, cache):
        self.manifest = read_json(manifest)
        self.rows = {r["id"]: r for r in self.manifest["records"]}
        if len(self.rows) != len(self.manifest["records"]):
            raise ValueError("Duplicate record IDs")
        self.root, self.cache = Path(root), Path(cache)

    @lru_cache(maxsize=32)
    def features(self, ident):
        r = self.rows[ident]
        fresh = feature_path(self.cache, ident)
        p = fresh if fresh.exists() else resolve(self.root, r["feature"]) if "feature" in r else fresh
        if not fresh.exists() and "clip_feature" in r:
            with np.load(resolve(self.root, r["clip_feature"]), allow_pickle=False) as f:
                c = f["clip"][:, :, :5].copy()
            with np.load(resolve(self.root, r["dino_feature"]), allow_pickle=False) as f:
                d = f["dino"].copy()
        else:
            with np.load(p, allow_pickle=False) as f:
                if "record_id" in f and str(f["record_id"].item()) != ident:
                    raise ValueError("Cache belongs to another record")
                if "rgb_sha" in f and r.get("rgb_sha") and list(f["rgb_sha"]) != r["rgb_sha"]:
                    raise ValueError("Cache image fingerprint mismatch")
                c, d = f["clip"].copy(), f["dino"].copy()
        if c.shape != (2, 4, 5, 512) or d.shape != (2, 32, 32, 384):
            raise ValueError(f"Invalid feature shape: {ident}")
        return c, d

    @lru_cache(maxsize=64)
    def mask(self, ident, key="mask"):
        return load_mask(resolve(self.root, self.rows[ident][key]))

    def truth(self, ident, family):
        row = self.rows[ident]
        change = self.mask(ident)
        valid = self.mask(ident, "valid") if "valid" in row else np.ones_like(change)
        if family in row.get("targets", {}):
            target = load_mask(resolve(self.root, row["targets"][family]))
        elif family in row.get("positive", []):
            target = change.copy()
        elif family in row.get("negative", []):
            target = np.zeros_like(change)
        else:
            raise ValueError(f"Uncertain/unlabeled {family} on {ident}; do not manufacture a negative")
        if target.shape != change.shape or valid.shape != change.shape:
            raise ValueError("Masks must share the native query resolution")
        return target, change, valid


def normalize_family(value):
    if isinstance(value, int):
        if not 1 <= value <= 4:
            raise ValueError("Family indices must lie in 1..4")
        return FAMILIES[value-1]
    if value not in FAMILIES:
        raise ValueError(f"Unsupported attribute family: {value}")
    return value


class Episodes(Dataset):
    def __init__(self, records, episodes):
        self.records, self.episodes = records, episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        e = self.episodes[index]
        family = normalize_family(e["family"])
        s, q = e["support"], e["query"]
        sr, qr = self.records.rows[s], self.records.rows[q]
        if sr["group"] == qr["group"]:
            raise ValueError("Support and query must have independent scene/original groups")
        mode = e.get("mode", "set")
        if mode not in ("set", "point", "mask"):
            raise ValueError(mode)
        cs, ds = self.records.features(s)
        cq, dq = self.records.features(q)
        if e.get("reverse_s", False):
            cs, ds = cs[::-1].copy(), ds[::-1].copy()
        if e.get("reverse_q", False):
            cq, dq = cq[::-1].copy(), dq[::-1].copy()
        sm, schange, sv = self.records.truth(s, family)
        target, change, valid = self.records.truth(q, family)
        if not sm.any():
            raise ValueError("The support must actually instantiate the requested attribute")
        point = np.full(2, -1, np.float32)
        if mode == "point":
            distance = distance_transform_edt(np.pad(sm, 1))[1:-1, 1:-1]
            y, x = np.unravel_index(distance.argmax(), sm.shape)
            point = np.asarray([(x+.5)/sm.shape[1], (y+.5)/sm.shape[0]], np.float32)
        method = e.get("mask_resize", "pil_nearest")
        selection = resize_mask(sm, method=method).astype(np.float32)
        return dict(clip=np.stack((cs, cq)), dino=np.stack((ds, dq)),
                    target=resize_mask(target, method=method).astype(np.float32), selection=selection,
                    valid=np.stack((resize_mask(sv, method=method), resize_mask(valid, method=method))).astype(np.float32),
                    change=np.stack((resize_mask(schange, method=method), resize_mask(change, method=method))).astype(np.float32),
                    point=point, support_mask=selection if mode == "mask" else np.zeros_like(selection),
                    mask_present=np.bool_(mode == "mask"))


class StaticFrames(Dataset):
    def __init__(self, records, frames):
        self.records, self.frames = records, frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        ident, time = self.frames[i]["record"], self.frames[i]["time"]
        c, d = self.records.features(ident)
        p = resolve(self.records.root, self.records.rows[ident]["static_labels"])
        with np.load(p, allow_pickle=False) as f:
            labels = f["labels"][time].copy()
        if labels.shape[0] != 4 or time not in (0, 1):
            raise ValueError("Static labels must be 2,4,H,W: color,size,shape,material")
        labels = np.stack([resize_mask(y, method="torch_nearest") for y in labels])
        return dict(clip=c[time], dino=d[time], labels=labels.astype(np.int64))


def validate_groups(manifest):
    groups, files = {}, {}
    for row in manifest["records"]:
        split = row["split"]
        if row["group"] in groups and groups[row["group"]] != split:
            raise ValueError("Scene/original group crosses splits")
        groups[row["group"]] = split
        for value in row.get("rgb_sha", []):
            if value in files and files[value] != split:
                raise ValueError("An exact RGB image crosses splits")
            files[value] = split
    return {"records": len(manifest["records"]), "groups": len(groups), "disjoint": True}

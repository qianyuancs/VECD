"""Native-resolution evaluation, source-only calibration, restartable shards."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Episodes, normalize_family
from .features import autocast
from .io import batch_to, model_inputs, read_json, sha256, write_json
from .metrics import confusion, measures


def summarize(rows):
    total = np.zeros(6, np.int64)
    groups = defaultdict(lambda: np.zeros(6, np.int64))
    for row in rows:
        total += row["counts"]
        groups[row["family"]] += row["counts"]
    positive = [r for r in rows if sum(r["counts"][i] for i in (0, 2)) > 0]
    empty = [r for r in rows if sum(r["counts"][i] for i in (0, 2)) == 0]
    return dict(requests=len(rows), pooled=measures(total),
                per_family={k: measures(v) for k, v in groups.items()},
                positive_requests=len(positive), empty_requests=len(empty),
                positive_only=measures(np.sum([r["counts"] for r in positive], axis=0)) if positive else None,
                empty_request_nonempty_rate=float(np.mean([r["counts"][1] > 0 for r in empty])) if empty else None,
                equal_family_foreground_IoU=float(np.mean([measures(v)["foreground_IoU"] for v in groups.values()])) if groups else 0.)


@torch.inference_mode()
def probability(model, item, shape, device):
    batch = batch_to([item], device)
    with autocast(device):
        output = model(**model_inputs(batch))
    p = F.interpolate(output["logits"][:, None].float().sigmoid(), shape,
                      mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite probability")
    return p


def evaluate(model, records, episodes, thresholds, device, output=None, shard=0, shards=1):
    model.eval()
    dataset = Episodes(records, episodes)
    rows = []
    if not 0 <= shard < shards:
        raise ValueError("Require 0 <= shard < shards")
    if not episodes:
        raise ValueError("No evaluation requests")
    for i in range(shard, len(episodes), shards):
        e = episodes[i]
        family = normalize_family(e["family"])
        mode = e.get("mode", "set")
        threshold = float(thresholds[mode])
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be in (0,1)")
        target, change, valid = records.truth(e["query"], family)
        dest = None if output is None else Path(output)/f"{i:08d}.json"
        if dest is not None and dest.exists():
            row = read_json(dest)
            if row["episode"] != e or row["threshold"] != threshold:
                raise ValueError("Do not mix different protocols in one output directory")
            path = dest.with_suffix(".npz")
            if sha256(path) != row["prediction_sha256"]:
                raise ValueError("Corrupt saved prediction")
            with np.load(path, allow_pickle=False) as z:
                pred = np.unpackbits(z["packed"], count=int(np.prod(z["shape"]))).reshape(z["shape"]).astype(bool)
            if confusion(pred, target, change, valid).tolist() != row["counts"]:
                raise ValueError("Saved mask/GT recount mismatch")
        else:
            p = probability(model, dataset[i], target.shape, device)
            pred = p >= threshold
            row = dict(index=i, episode=e, family=family, threshold=threshold,
                       counts=confusion(pred, target, change, valid).tolist())
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                path = dest.with_suffix(".npz")
                np.savez_compressed(path, packed=np.packbits(pred), shape=np.asarray(pred.shape))
                row["prediction_sha256"] = sha256(path)
                write_json(dest, row)
        rows.append(row)
    summary = summarize(rows)
    if output is not None:
        write_json(Path(output)/f"summary-shard-{shard}.json", summary)
    return summary


def merge_shards(records, episodes, output):
    rows = []
    for i, e in enumerate(episodes):
        dest = Path(output)/f"{i:08d}.json"
        row = read_json(dest)
        if row["index"] != i or row["episode"] != e:
            raise ValueError("Request ordering mismatch")
        path = dest.with_suffix(".npz")
        if sha256(path) != row["prediction_sha256"]:
            raise ValueError("Prediction hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            pred = np.unpackbits(z["packed"], count=int(np.prod(z["shape"]))).reshape(z["shape"]).astype(bool)
        if confusion(pred, *records.truth(e["query"], normalize_family(e["family"]))).tolist() != row["counts"]:
            raise ValueError("Independent mask recount failed")
        rows.append(row)
    summary = summarize(rows)
    summary["all_saved_masks_recounted"] = True
    if (Path(output)/"binding.json").exists():
        summary["binding"] = read_json(Path(output)/"binding.json")
    write_json(Path(output)/"results.json", summary)
    return summary


@torch.inference_mode()
def calibrate(model, records, episodes, device):
    if not episodes:
        raise ValueError("No source-calibration requests")
    if any(records.rows[e["query"]]["split"] != "cal" or records.rows[e["support"]]["split"] != "cal" for e in episodes):
        raise ValueError("Calibration may only use source-calibration support/query records")
    grid = np.arange(.05, 1., .05)
    model.eval()
    dataset, rows = Episodes(records, episodes), []
    for i, e in enumerate(episodes):
        family = normalize_family(e["family"])
        target, change, valid = records.truth(e["query"], family)
        p = probability(model, dataset[i], target.shape, device)
        for threshold in grid:
            rows.append(dict(mode=e.get("mode", "set"), family=family, threshold=float(threshold),
                             counts=confusion(p >= threshold, target, change, valid).tolist()))
    thresholds = {}
    for mode in sorted({r["mode"] for r in rows}):
        scores = [(float(t), summarize([r for r in rows if r["mode"] == mode and r["threshold"] == t])["equal_family_foreground_IoU"]) for t in grid]
        thresholds[mode] = max(scores, key=lambda x: (x[1], -x[0]))[0]
    return dict(thresholds=thresholds, source_calibration_only=True, target_tests_used=False,
                criterion="equal-family pooled foreground IoU; smallest threshold breaks ties")

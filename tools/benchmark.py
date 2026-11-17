"""Evaluate the same user-supplied requests for each training seed."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from subprocess import run
import sys


def aggregate(rows):
    if len(rows) < 2 or not all(r["all_saved_masks_recounted"] for r in rows):
        raise ValueError("Expected at least two complete runs")
    if len({(r["requests"], r["binding"]["manifest_sha256"], r["binding"]["episodes_sha256"])
            for r in rows}) != 1:
        raise ValueError("Seeds were evaluated on different protocols")
    if len({r["binding"]["weights_sha256"] for r in rows}) != len(rows):
        raise ValueError("Repeated checkpoint")
    metrics = {}
    for key in ("binary_mIoU", "F1"):
        scores = [100 * r["pooled"][key] for r in rows]
        metrics[key] = {"mean_percent": mean(scores), "sample_sd_points": stdev(scores)}
    return metrics


def vecd(command, *args):
    run([sys.executable, "-m", "vecd", command, *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("protocol", type=Path, help="Directory containing records.json and episodes.json")
    p.add_argument("--seeds", type=int, nargs="+", choices=(24201, 24202, 24203), default=[24201, 24202, 24203])
    p.add_argument("--data-root", default="data/prepared")
    p.add_argument("--cache", default="cache/features")
    p.add_argument("--cache-features", action="store_true", help="Extract missing features from RGB files")
    p.add_argument("--trained", action="store_true", help="Use locally trained and calibrated checkpoints")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Each seed must be unique")
    folder = args.protocol.resolve()
    for filename in ("records.json", "episodes.json"):
        if not (folder / filename).is_file():
            p.error(f"Missing {folder / filename}")
    data = ["--manifest", folder/"records.json", "--data-root", args.data_root, "--cache", args.cache]
    episodes = ["--episodes", folder/"episodes.json"]
    group = "trained" if args.trained else "released"
    out = Path("outputs/benchmarks") / group / folder.name
    runs = []
    models = []
    for seed in args.seeds:
        if args.trained:
            checkpoint = Path(f"outputs/mixed_{seed}/best.safetensors")
            config = checkpoint.parent / "calibrated.json"
        else:
            checkpoint = Path(f"weights/{seed}/model.safetensors")
            config = Path(f"configs/inference/seed{seed}.json")
        for f in (checkpoint, config):
            if not f.is_file():
                p.error(f"Missing {f}")
        models.append((seed,checkpoint,config))
    if args.cache_features:
        vecd("cache", *data, "--device", args.device)
    for seed, checkpoint, config in models:
        destination = out / str(seed)
        vecd("evaluate", *data, *episodes, "--output", destination, "--weights", checkpoint,
             "--model-config", config, "--device", args.device)
        vecd("merge", *data, *episodes, "--output", destination)
        runs.append(json.loads((destination/"results.json").read_text(encoding="utf8")))
    if len(runs) > 1:
        result = dict(seeds=args.seeds, requests_per_seed=runs[0]["requests"], metrics=aggregate(runs))
        rendered = json.dumps(result, indent=2)
        (out/("summary_"+"_".join(map(str,args.seeds))+".json")).write_text(rendered+"\n", encoding="utf8")
        print(rendered)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import Records, validate_groups
from .features import FrozenFeatures, autocast, cache_records
from .io import load_mask, load_weights, read_json, resize_mask, sha256, write_json
from .model import VECD


def model_args(parser):
    parser.add_argument("--weights", required=True, help="Standalone model.safetensors")
    parser.add_argument("--device", default="cuda")


def data_args(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache", required=True)


def backbone_args(parser):
    from .features import CLIP_ID, DINO_ID
    parser.add_argument("--clip-model", default=CLIP_ID)
    parser.add_argument("--dino-model", default=DINO_ID)
    parser.add_argument("--offline", action="store_true")


def infer(args):
    device = torch.device(args.device)
    model = load_weights(VECD(), args.weights, device).eval()
    extractor = FrozenFeatures(args.clip_model, args.dino_model, device, args.offline)
    with torch.inference_mode():
        c, d = zip(extractor.pair(args.example_before, args.example_after),
                   extractor.pair(args.query_before, args.query_after))
        # zip above groups CLIP and DINO tensors, not image-pair axes.
        clip = torch.stack(c)[None].float()
        dino = torch.stack(d)[None].float()
        point = None if args.point is None else torch.tensor([args.point], device=device)
        support_mask = None if args.example_mask is None else torch.tensor(
            resize_mask(load_mask(args.example_mask)), device=device)[None].float()
        with autocast(device):
            out = model(clip, dino, point=point, support_mask=support_mask)
        with Image.open(args.query_before) as image:
            shape = image.height, image.width
        prob = F.interpolate(out["logits"][:, None].float().sigmoid(), shape,
                              mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    mode = "mask" if args.example_mask else "point" if args.point else "set"
    config = read_json(args.model_config)
    if "model_sha256" in config and sha256(args.weights) != config["model_sha256"]:
        raise ValueError("Model/config fingerprint mismatch")
    threshold = config["thresholds"][mode]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prob >= threshold).astype(np.uint8)*255).save(output/"prediction.png")
    np.save(output/"probability.npy", prob)
    write_json(output/"inference.json", dict(mode=mode, threshold=threshold,
               weights_sha256=sha256(args.weights), query_GT_used=False, output_shape=list(shape)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vecd")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("infer")
    model_args(p); backbone_args(p)
    p.add_argument("--example-before", "--support-before", dest="example_before", required=True)
    p.add_argument("--example-after", "--support-after", dest="example_after", required=True)
    for name in ("query-before", "query-after", "model-config", "output"):
        p.add_argument("--"+name, required=True)
    cues = p.add_mutually_exclusive_group()
    cues.add_argument("--example-mask", "--support-mask", dest="example_mask",
                      help="Optional annotated-example reproduction; NOT the unannotated main setting")
    cues.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"),
                      help="Optional example cue retained for training-recipe reproduction")
    p = sub.add_parser("cache")
    data_args(p); backbone_args(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, help="Stop early for a smoke test; not a full training result")
    for command in ("evaluate", "calibrate", "merge"):
        p = sub.add_parser(command)
        data_args(p)
        p.add_argument("--episodes", required=True)
        p.add_argument("--output", required=True)
        if command != "merge":
            model_args(p)
        if command == "evaluate":
            p.add_argument("--model-config", required=True)
            p.add_argument("--shard", type=int, default=0); p.add_argument("--shards", type=int, default=1)
    p = sub.add_parser("validate")
    p.add_argument("--manifest", required=True)
    for command in ("prepare-annotated", "prepare-natural"):
        p = sub.add_parser(command)
        p.add_argument("--metadata", required=True); p.add_argument("--raw-root", required=True)
        p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.command == "infer":
        infer(args)
    elif args.command == "train":
        from .training import train
        print(json.dumps(train(args.config, args.device, args.max_steps)))
    elif args.command == "cache":
        e = FrozenFeatures(args.clip_model, args.dino_model, args.device, args.offline)
        cache_records(read_json(args.manifest), args.data_root, Path(args.cache), e, args.shard, args.shards, args.overwrite)
    elif args.command == "validate":
        print(json.dumps(validate_groups(read_json(args.manifest))))
    elif args.command.startswith("prepare-"):
        from .preparation import prepare_annotated_pairs, prepare_natural
        fn = prepare_annotated_pairs if args.command == "prepare-annotated" else prepare_natural
        fn(args.metadata, args.raw_root, args.output)
    else:
        from .evaluation import evaluate, calibrate, merge_shards
        records = Records(args.manifest, args.data_root, args.cache)
        episodes = read_json(args.episodes)
        if args.command == "merge":
            print(json.dumps(merge_shards(records, episodes, args.output)))
            return
        device = torch.device(args.device)
        model = load_weights(VECD(), args.weights, device).eval()
        if args.command == "calibrate":
            value = calibrate(model, records, episodes, device)
            value["model_sha256"] = sha256(args.weights)
            write_json(args.output, value)
            print(json.dumps(value))
        else:
            cfg = read_json(args.model_config)
            if cfg.get("model_sha256", sha256(args.weights)) != sha256(args.weights):
                raise ValueError("Weights/config SHA mismatch")
            output = Path(args.output)
            binding = dict(weights_sha256=sha256(args.weights), manifest_sha256=sha256(args.manifest),
                           episodes_sha256=sha256(args.episodes), thresholds=cfg["thresholds"])
            if (output/"binding.json").exists() and read_json(output/"binding.json") != binding:
                raise ValueError("Output directory contains results of another model/protocol")
            write_json(output/"binding.json", binding)
            print(json.dumps(evaluate(model, records, episodes, cfg["thresholds"], device, output, args.shard, args.shards)))


if __name__ == "__main__":
    main()

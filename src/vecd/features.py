"""Frozen visual encoders with the exact retained image preprocessing."""
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .io import load_rgb, sha256, write_json

CLIP_ID = "openai/clip-vit-base-patch16"
DINO_ID = "facebook/dinov2-small"
PIPELINE = "clip-4tiles-cls-4quadrants-dino448-bicubic-v1"


def autocast(device, enabled=True):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and enabled else nullcontext()


class FrozenFeatures(nn.Module):
    def __init__(self, clip_model=CLIP_ID, dino_model=DINO_ID, device="cuda", offline=False):
        super().__init__()
        from transformers import CLIPModel, AutoModel
        self.clip = CLIPModel.from_pretrained(clip_model, local_files_only=offline)
        self.dino = AutoModel.from_pretrained(dino_model, local_files_only=offline, use_safetensors=True)
        self.requires_grad_(False).eval().to(device)
        self.register_buffer("cm", torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None])
        self.register_buffer("cs", torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None])
        self.register_buffer("dm", torch.tensor([.485, .456, .406], device=device)[None, :, None, None])
        self.register_buffer("ds", torch.tensor([.229, .224, .225], device=device)[None, :, None, None])
        self.identities = {"clip": str(clip_model), "dino": str(dino_model), "pipeline": PIPELINE}

    @torch.inference_mode()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 512, 512) or rgb.dtype != torch.uint8:
            raise ValueError("Expected uint8 N,3,512,512 RGB")
        device = self.cm.device
        pixels = rgb.to(device).float()/255
        crops = torch.stack([pixels[t, :, y:y+256, x:x+256]
                             for t in range(len(rgb)) for y in (0, 256) for x in (0, 256)])
        with autocast(device):
            x = F.interpolate(crops, (224, 224), mode="bicubic", align_corners=False, antialias=True)
            v = self.clip.vision_model((x-self.cm)/self.cs, return_dict=True)
            cls = self.clip.visual_projection(v.pooler_output)
            patch = v.last_hidden_state[:, 1:].reshape(-1, 14, 14, 768)
            pools = torch.stack([patch[:, y:y+7, x:x+7].mean((1, 2)) for y in (0, 7) for x in (0, 7)], 1)
            pools = self.clip.visual_projection(self.clip.vision_model.post_layernorm(pools))
            c = torch.cat((cls[:, None], pools), 1).reshape(len(rgb), 4, 5, 512)
            x = F.interpolate(pixels, (448, 448), mode="bicubic", align_corners=False, antialias=True)
            d = self.dino(pixel_values=(x-self.dm)/self.ds).last_hidden_state[:, 1:]
            d = d.reshape(len(rgb), 32, 32, 384)
        # Historical paired training consumes float16 cached features.
        return c.float().to(torch.float16), d.float().to(torch.float16)

    @torch.inference_mode()
    def pair(self, before, after):
        return self(torch.from_numpy(np.stack((load_rgb(before), load_rgb(after)))))


def feature_path(cache, ident):
    import hashlib
    return cache / (hashlib.sha256(str(ident).encode()).hexdigest()+".npz")


def cache_records(manifest, root, cache, extractor, shard=0, shards=1, overwrite=False):
    from pathlib import Path
    from .io import resolve
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    from .io import read_json
    identity = cache/"extractor.json"
    if identity.exists() and read_json(identity) != extractor.identities:
        raise ValueError("Backbone/preprocessing changed; use a separate cache directory")
    write_json(identity, extractor.identities)
    if not (0 <= shard < shards):
        raise ValueError("Require 0 <= shard < shards")
    for r in sorted(manifest["records"], key=lambda r: r["id"])[shard::shards]:
        dest = feature_path(cache, r["id"])
        inputs = [resolve(root, r[k]) for k in ("before", "after")]
        hashes = [sha256(p) for p in inputs]
        if dest.exists() and not overwrite:
            with np.load(dest, allow_pickle=False) as old:
                if list(old["file_sha"]) != hashes:
                    raise ValueError(f"Stale image cache for {r['id']}")
            continue
        c, d = extractor.pair(*inputs)
        temp = dest.with_suffix(".tmp.npz")
        np.savez(temp, clip=c.cpu().numpy(), dino=d.cpu().numpy(), file_sha=np.asarray(hashes),
                 record_id=np.asarray(r["id"]), rgb_sha=np.asarray(r.get("rgb_sha", [])))
        temp.replace(dest)
    write_json(cache/f"extractor-shard-{shard}.json", dict(**extractor.identities, shard=shard, shards=shards))

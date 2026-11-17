import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4*1024**2), b""):
            h.update(data)
    return h.hexdigest()


def resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else Path(root)/p


def load_rgb(path, size=512):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)).transpose(2, 0, 1).copy()


def load_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def resize_mask(mask, size=512, method="pil_nearest"):
    if method == "torch_nearest":
        from torch.nn import functional as F
        x = torch.from_numpy(np.asarray(mask, np.float32).copy())[None, None]
        return F.interpolate(x, (size, size), mode="nearest")[0, 0].byte().numpy()
    if method != "pil_nearest":
        raise ValueError("Unknown registered label-resize convention")
    return np.asarray(Image.fromarray(np.asarray(mask, np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST)).copy()


def load_weights(model, path, device="cpu"):
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.to(device)


def save_weights(model, path):
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.safetensors")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp),
              metadata={"architecture": "vecd-uniform-512-no-rgb", "format": "0.1"})
    temp.replace(path)


def batch_to(items, device):
    return {key: torch.as_tensor(np.stack([item[key] for item in items]), device=device)
            for key in items[0]}


def model_inputs(batch):
    return dict(clip=batch["clip"], dino=batch["dino"], point=batch["point"],
                support_mask=batch["support_mask"], mask_present=batch["mask_present"])

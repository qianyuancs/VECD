"""Static and paired training with explicit initialization and exact resume state."""
import copy
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Episodes, Records, StaticFrames, validate_groups
from .evaluation import evaluate
from .features import autocast
from .io import batch_to, load_weights, model_inputs, read_json, save_weights, sha256, write_json
from .losses import paired_loss, static_loss
from .model import VECD


def keep_list(items):
    return items


def setup(seed, threads=2):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def state_digest(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def event(output, **values):
    row = dict(time=time.time(), **values)
    write_json(output/"status.json", row)
    import json
    with (output/"train.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(row)+"\n")
    print(json.dumps(row), flush=True)


def checkpoint(path, model, optimizer, **values):
    temp = path.with_suffix(".tmp.pt")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(), **values), temp)
    temp.replace(path)


def train(config, device="cuda", max_steps=None):
    cfg = read_json(config)
    device = torch.device(device)
    setup(cfg["seed"], cfg.get("threads", 2))
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, config_sha256=sha256(config), initialization_sha256=sha256(cfg["initialization"]),
                    input_sha256={k:sha256(cfg[k]) for k in ("manifest","episodes","frames","prototypes","validation") if cfg.get(k)})
    if (output/"run.json").exists() and read_json(output/"run.json") != identity:
        raise ValueError("Existing output belongs to another training configuration")
    write_json(output/"run.json", identity)
    records = Records(cfg["manifest"], cfg["data_root"], cfg["cache"])
    validate_groups(records.manifest)
    model = load_weights(VECD(), cfg["initialization"], device)
    static = cfg["stage"] == "static"
    if cfg["stage"] not in ("static", "parent", "mixed"):
        raise ValueError("stage must be static, parent or mixed")
    teacher = None
    if static:
        teacher = copy.deepcopy(model.base.state).eval().requires_grad_(False)
        model.requires_grad_(False)
        model.base.state.requires_grad_(True)
        from safetensors.torch import load_file
        values = load_file(cfg["prototypes"])
        prototypes = [values[f"prototype_{i}"] for i in range(4)]
        entries = read_json(cfg["frames"])
        for e in entries:
            if records.rows[e["record"]]["split"] != "train":
                raise ValueError("Static training frame outside train split")
    else:
        entries = read_json(cfg["episodes"])
        excluded = set(cfg.get("heldout_families", []))
        for e in entries:
            from .data import normalize_family
            if records.rows[e["support"]]["split"] != "train" or records.rows[e["query"]]["split"] != "train":
                raise ValueError("Paired training uses non-training support/query")
            # Exclusion covers all actual support/query changes, not just requested labels.
            for key in ("support", "query"):
                if excluded.intersection(records.rows[e[key]].get("positive", [])):
                    raise ValueError("Held-out family leaked through a positive or negative pair")
            if normalize_family(e["family"]) in excluded:
                raise ValueError("Held-out requested family leaked")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", .001))
    frozen = state_digest(model.base.state.state_dict())
    start_epoch = cursor = step = 0
    best = -1.
    latest = output/"resume.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        for value in optimizer.state.values():
            for k, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    value[k] = tensor.to(device)
        restore_rng(ck["rng"])
        start_epoch, cursor, step, best = ck["epoch"], ck["cursor"], ck["step"], ck["best"]
        # Always preserve the original initial teacher, not a resumed student.
    effective = cfg.get("effective_batch", 4 if static else 8)
    micro = cfg.get("microbatch", 1 if static else 2)
    if micro < 1 or effective % micro:
        raise ValueError("microbatch must divide effective_batch")
    workers = cfg.get("workers", 2)
    started = time.monotonic()
    for epoch in range(start_epoch, cfg["epochs"]):
        current = entries if static else [e for e in entries if e.get("epoch", 0) == epoch]
        if not current:
            raise ValueError(f"No entries for epoch {epoch}")
        if cfg.get("shuffle", static):
            shuffle_seed = cfg["seed"]+epoch if static else cfg["seed"]*1000+epoch
            order = np.random.default_rng(shuffle_seed).permutation(len(current)).tolist()
        else:
            order = list(range(len(current)))
        ds = StaticFrames(records, current) if static else Episodes(records, current)
        batches = [order[i:i+effective] for i in range(cursor, len(order), effective)]
        loader_kw = dict(dataset=ds, batch_sampler=batches, num_workers=workers, collate_fn=keep_list)
        if workers:
            loader_kw.update(prefetch_factor=2, persistent_workers=True)
        loader = DataLoader(**loader_kw)
        model.train()
        for items in loader:
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            if static:
                # Match per-attribute CE denominators over the full effective batch.
                from torch.nn import functional as F
                all_labels = torch.as_tensor(np.stack([r["labels"] for r in items]), device=device)
                all_labels = F.interpolate(all_labels.float(), (64, 64), mode="nearest").long()
                denom = (all_labels != 255).sum((0, 2, 3)).clamp_min(1)
            for start in range(0, len(items), micro):
                part = items[start:start+micro]
                batch = batch_to(part, device)
                with autocast(device):
                    if static:
                        from torch.nn import functional as F
                        z = model.base.state(batch["clip"].float(), batch["dino"].float())
                        with torch.no_grad():
                            ref = teacher(batch["clip"].float(), batch["dino"].float())
                        loss = z.sum()*0
                        for local, facet in enumerate((1, 4, 3, 2)):
                            proto = F.normalize(prototypes[local].float().to(device), dim=-1)
                            logits = (z[:, :, facet].float() @ proto.T/.07).transpose(1, 2)
                            logits = logits.reshape(len(part), len(proto), 64, 64)
                            loss = loss+F.cross_entropy(logits, all_labels[start:start+len(part), local],
                                ignore_index=255, reduction="sum")/denom[local]/4
                        loss = loss+.25*(1-(z.float()*ref.float()).sum(-1)).mean()*len(part)/len(items)
                    else:
                        loss, _ = paired_loss(model(**model_inputs(batch)), batch)
                        loss = loss*len(part)/len(items)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            step, cursor = step+1, cursor+len(items)
            if step % cfg.get("save_every", 50) == 0 or cursor == len(current) or (max_steps is not None and step >= max_steps):
                checkpoint(latest, model, optimizer, epoch=epoch, cursor=cursor, step=step, best=best)
            event(output, stage=cfg["stage"], epoch=epoch+1, epochs=cfg["epochs"], step=step,
                  cursor=cursor, loss=total, gradient_norm=float(norm), seconds=time.monotonic()-started)
            if max_steps is not None and step >= max_steps:
                return {"stopped_after_smoke_budget": True, "step": step}
        cursor = 0
        if not static and cfg.get("validation") and (epoch+1) % cfg.get("validate_every", 2) == 0:
            validation = read_json(cfg["validation"])
            if any(records.rows[e[k]]["split"] != "val" for e in validation for k in ("support", "query")):
                raise ValueError("Checkpoint selection must only use source validation")
            result = evaluate(model, records, validation, {"set": .5, "point": .5, "mask": .5}, device)
            score = result["equal_family_foreground_IoU"]
            event(output, stage="source_validation", epoch=epoch+1, score=score)
            if score > best:
                best = score
                save_weights(model, output/"best.safetensors")
                write_json(output/"selection.json", dict(epoch=epoch+1, score=score, source_only=True))
        if not static and state_digest(model.base.state.state_dict()) != frozen:
            raise AssertionError("Frozen state reader changed")
        checkpoint(latest, model, optimizer, epoch=epoch+1, cursor=0, step=step, best=best)
    save_weights(model, output/"final.safetensors")
    write_json(output/"complete.json", dict(step=step, stage=cfg["stage"], epochs=cfg["epochs"],
               final_sha256=sha256(output/"final.safetensors"), source_only_selection=True))
    return read_json(output/"complete.json")

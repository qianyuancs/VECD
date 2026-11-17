import torch
from torch.nn import functional as F


def segmentation_loss(logits, target, valid):
    logits, target, valid = logits.float(), target.float(), valid.float()
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("Loss shapes must agree")
    dims = tuple(range(1, logits.ndim))
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none")*valid).sum(dims)
    bce = bce/valid.sum(dims).clamp_min(1)
    p, y = logits.sigmoid()*valid, target*valid
    dice = 1-(2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)
    return (bce+dice).mean()


def paired_loss(output, batch):
    losses, terms = [], []
    for i in range(len(batch["target"])):
        sl = slice(i, i+1)
        query = segmentation_loss(output["logits"][sl], batch["target"][sl], batch["valid"][sl, 1])
        change = segmentation_loss(output["change"][sl].flatten(0, 1),
                                   batch["change"][sl].flatten(0, 1), batch["valid"][sl].flatten(0, 1))
        selection = output["selection"][sl].sum()*0 if batch["mask_present"][i] else segmentation_loss(
            output["selection"][sl], batch["selection"][sl], batch["valid"][sl, 0])
        losses.append(query+.5*selection+.25*change)
        terms.append(torch.stack((query.detach(), selection.detach(), change.detach())))
    return torch.stack(losses).mean(), torch.stack(terms).mean(0)


def static_loss(states, teacher, labels, prototypes, temperature=.07):
    labels = F.interpolate(labels.float(), (64, 64), mode="nearest").long()
    losses = []
    for local, facet in enumerate((1, 4, 3, 2)):
        proto = F.normalize(prototypes[local].float().to(states.device), dim=-1)
        logits = (states[:, :, facet].float() @ proto.T / temperature).transpose(1, 2)
        logits = logits.reshape(len(states), len(proto), 64, 64)
        losses.append(F.cross_entropy(logits, labels[:, local], ignore_index=255))
    retain = (1-(states.float()*teacher.float()).sum(-1)).mean()
    return torch.stack(losses).mean()+.25*retain

"""Uniform support reading with optional support-space cues.

Feature axes: B, pair(support/query), time(before/after), ... .
No operation labels, class IDs, captions or query masks enter forward.
The parameter names intentionally preserve the released checkpoint contract.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

FACETS = ("category", "color", "material", "shape", "size")
FAMILIES = ("color", "size", "shape", "material")
STATE_SIZE, MATCH_SIZE = 64, 32


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU())


class StateReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_projection = nn.Linear(512, 512)
        self.dino_projection = nn.Linear(384, 512)
        self.query_projection = nn.Linear(512, 512)
        self.position = nn.Parameter(torch.zeros(6, 512))
        self.attention = nn.MultiheadAttention(512, 4, dropout=0, batch_first=True)
        self.ln = nn.LayerNorm(512)
        self.ffn = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512))
        self.output = nn.Linear(512, 512)

    def forward(self, clip_tokens, dino_state, queries):
        shape = clip_tokens.shape[:-2]
        c = F.normalize(clip_tokens.float().reshape(-1, 5, 512), dim=-1, eps=1e-8)
        d = F.normalize(dino_state.float().reshape(-1, 384), dim=-1, eps=1e-8)
        memory = torch.cat((self.clip_projection(c), self.dino_projection(d)[:, None]), 1)
        memory = memory + self.position
        q = self.query_projection(F.normalize(queries.float(), dim=-1, eps=1e-8))
        q = q[None].expand(len(c), -1, -1)
        x = q + self.attention(q, memory, memory, need_weights=False)[0]
        x = x + self.ffn(self.ln(x))
        return F.normalize(c[:, :1] + self.output(x), dim=-1, eps=1e-8).reshape(
            *shape, len(queries), 512)


class StateField(nn.Module):
    def __init__(self):
        super().__init__()
        self.reader = StateReader()
        self.register_buffer("queries", torch.zeros(5, 512))

    def forward(self, clip, dino):
        if clip.shape[1:] != (4, 5, 512) or dino.shape[1:] != (32, 32, 384):
            raise ValueError("Expected CLIP B,4,5,512 and DINO B,32,32,384")
        axis = torch.arange(32, device=clip.device) // 16
        tile = (axis[:, None] * 2 + axis[None]).flatten()
        with torch.autocast(device_type=clip.device.type, enabled=False):
            z = self.reader(clip[:, tile].float(), dino.reshape(-1, 1024, 384).float(),
                            self.queries.float())
        b = len(clip)
        z = resize(z.permute(0, 2, 3, 1).reshape(b, 5*512, 32, 32), (64, 64))
        return F.normalize(z.reshape(b, 5, 512, 4096).permute(0, 3, 1, 2), dim=-1, eps=1e-8)


class PairGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.Conv2d(768, 24, 1), nn.GroupNorm(8, 24), nn.GELU())
        # First 24 inputs are explicitly zero: there is no RGB CNN.
        self.fuse = block(48, 24)
        self.change = nn.Conv2d(24, 1, 1)

    def forward(self, dino, image_size):
        a, b = dino.movedim(-1, -3).unbind(1)
        context = self.context(torch.cat(((a+b)*.5, (a-b).abs()), 1).float())
        context = resize(context, tuple((n+3)//4 for n in image_size))
        x = self.fuse(torch.cat((torch.zeros_like(context), context), 1))
        return x, self.change(x)[:, 0]


def pool_states(z, size=MATCH_SIZE):
    b, t, n, k, d = z.shape
    side = math.isqrt(n)
    if side*side != n:
        raise ValueError("State positions must form a square grid")
    a = z.permute(0, 1, 3, 4, 2).reshape(b*t, k*d, side, side)
    a = F.adaptive_avg_pool2d(a, (size, size))
    return F.normalize(a.reshape(b, t, k, d, size*size).permute(0, 1, 4, 2, 3), dim=-1)


def family_evidence(query, support, weight):
    qe = (query[:, 1].float()-query[:, 0].float()).square().sum(-1)
    se = (support[:, 1].float()-support[:, 0].float()).square().sum(-1)
    mass = weight.float().clamp_min(0)
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)
    s = (se*mass[..., None]).sum(1)
    sa = s / s.sum(-1, keepdim=True).clamp_min(1e-8)
    qa = qe / qe.sum(-1, keepdim=True).clamp_min(1e-8)
    active = (se*mass[..., None]).sum((1, 2)) > 1e-8
    ss = sa[:, None].expand_as(qe)
    signal = torch.sqrt(qe.clamp_min(0)*s[:, None].clamp_min(0))
    match = qa*ss*active[:, None, None]
    # Duplicate interaction is part of the trained six-channel interface.
    return torch.stack((match, qa, ss, signal, match), -1).permute(0, 2, 3, 1)


def mask_selection(mask, size, batch):
    if mask.ndim != 3 or len(mask) != batch:
        raise ValueError("support_mask must be B,H,W")
    if not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all():
        raise ValueError("support_mask must be finite and within [0,1]")
    return torch.logit(F.interpolate(mask[:, None].float(), size, mode="area")[:, 0])


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = StateField()
        self.geometry = PairGeometry()
        self.selector = nn.Sequential(block(32, 32), nn.Conv2d(32, 1, 1))
        self.facet_readout = nn.Sequential(block(6, 16), nn.Conv2d(16, 8, 1))
        self.decoder = nn.Sequential(block(41, 32), block(32, 16), nn.Conv2d(16, 1, 1))

    def encode(self, clip, dino, image_size=(512, 512)):
        clip, dino = clip.float(), dino.float()
        b, t = clip.shape[:2]
        if t != 2:
            raise ValueError("A pair must contain before and after")
        with torch.no_grad():
            z = self.state(clip.flatten(0, 1), dino.flatten(0, 1)).reshape(b, 2, -1, 5, 512)
        g, c = self.geometry(dino, image_size)
        return z, g, c

    def point_features(self, states, point):
        b, _, n, k, d = states.shape
        side = math.isqrt(n)
        active = (point >= 0).all(-1)
        coords = point.clamp(0, 1)*2-1
        fields = states.permute(0, 1, 3, 4, 2).reshape(b*2, k*d, side, side)
        grid = coords[:, None, None, None].expand(-1, 2, -1, -1, -1).reshape(b*2, 1, 1, 2)
        anchor = F.grid_sample(fields, grid, mode="bilinear", padding_mode="border", align_corners=False)
        anchor = anchor.reshape(b, 2, k, d)
        a, after = states[:, 0], states[:, 1]
        ca = (a*anchor[:, None, 0]).sum(-1)
        cb = (after*anchor[:, None, 1]).sum(-1)
        cd = (F.normalize(after-a, dim=-1)*F.normalize(anchor[:, 1]-anchor[:, 0], dim=-1)[:, None]).sum(-1)
        changed = (anchor[:, 1]-anchor[:, 0]).square().sum(-1)
        weights = changed/changed.sum(-1, keepdim=True).clamp_min(1e-8)
        relation = torch.stack([(x*weights[:, None]).sum(-1) for x in (ca, cb, cd)], 1)
        relation = relation.reshape(b, 3, side, side)*active[:, None, None, None]
        axis = (torch.arange(side, device=states.device)+.5)/side
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        distance = (xx[None]-point[:, 0, None, None]).square()+(yy[None]-point[:, 1, None, None]).square()
        gaussian = torch.exp(-distance/(2*.035**2))*active[:, None, None]
        extras = torch.stack((gaussian, active[:, None, None].expand_as(gaussian).float(),
                              torch.zeros_like(gaussian)), 1)
        return torch.cat((relation, extras), 1)

    def select(self, states, geometry, point):
        energy = (states[:, 1]-states[:, 0]).square().sum(-1).transpose(1, 2).reshape(-1, 5, 64, 64)
        prompt = self.point_features(states, point)
        energy = torch.stack((energy.mean(1), energy.amax(1)), 1)
        return self.selector(torch.cat((geometry, resize(energy, geometry.shape[-2:]),
                                        resize(prompt, geometry.shape[-2:])), 1))[:, 0]

    def decode(self, query_states, support_states, geometry, change, selection):
        q, s = pool_states(query_states), pool_states(support_states)
        weight = F.adaptive_avg_pool2d(selection[:, None].sigmoid(), (32, 32)).flatten(1)
        cost = family_evidence(q, s, weight).reshape(len(q), 5, 5, 32, 32)
        energy = (q[:, 1]-q[:, 0]).square().sum(-1).transpose(1, 2).reshape(len(q), 5, 32, 32)
        hidden = self.facet_readout(torch.cat((cost, energy[:, :, None]), 2).flatten(0, 1))
        hidden = hidden.reshape(len(q), 5, 8, 32, 32)
        w = cost[:, :, 4]
        w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
        combined = torch.cat(((hidden*w[:, :, None]).sum(1), hidden.amax(1)), 1)
        context = torch.cat((geometry, resize(combined, geometry.shape[-2:]), change[:, None].sigmoid()), 1)
        return self.decoder(context)[:, 0], cost


class VECD(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = Core()
        self.base.state.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.state.eval()
        return self

    def select(self, states, geometry, point, support_mask=None, mask_present=None):
        b = len(states)
        if mask_present is None:
            mask_present = torch.full((b,), support_mask is not None, dtype=torch.bool, device=states.device)
        if mask_present.shape != (b,) or mask_present.dtype != torch.bool:
            raise ValueError("mask_present must be a boolean B-vector")
        if mask_present.any() and support_mask is None:
            raise ValueError("A present mask requires support_mask")
        points = torch.where(mask_present[:, None], torch.full_like(point, -1), point)
        learned = self.base.select(states, geometry, points)
        selected = learned
        if mask_present.any():
            given = mask_selection(support_mask, geometry.shape[-2:], b)
            selected = torch.where(mask_present[:, None, None], given, selected)
        unprompted = (point < 0).all(-1) & ~mask_present
        selected = torch.where(unprompted[:, None, None], torch.zeros_like(selected), selected)
        return selected, learned, mask_present

    def forward(self, clip, dino, point=None, support_mask=None, mask_present=None, image_size=(512, 512)):
        if clip.ndim != 6 or clip.shape[1:3] != (2, 2):
            raise ValueError("CLIP axes must be B,support/query,before/after,4,5,512")
        b = len(clip)
        if point is None:
            point = torch.full((b, 2), -1., device=clip.device)
        if point.shape != (b, 2) or not torch.isfinite(point).all():
            raise ValueError("point must be finite B,2 normalized support coordinates")
        absent = (point == -1).all(-1)
        if not ((absent | ((point >= 0) & (point <= 1)).all(-1)).all()):
            raise ValueError("Use [-1,-1] for no point or [x,y] within [0,1]")
        z, g, c = self.base.encode(clip.flatten(0, 1), dino.flatten(0, 1), image_size)
        z, g, c = [x.reshape(b, 2, *x.shape[1:]) for x in (z, g, c)]
        selected, learned, present = self.select(z[:, 0], g[:, 0], point, support_mask, mask_present)
        logits, costs = self.base.decode(z[:, 1], z[:, 0], g[:, 1], c[:, 1], selected)
        if support_mask is not None:
            active = ~present | (support_mask.flatten(1).sum(1) > 0)
            logits = torch.where(active[:, None, None], logits, torch.full_like(logits, -20))
        return dict(logits=resize(logits[:, None], image_size)[:, 0],
                    selection=resize(learned[:, None], image_size)[:, 0], costs=costs,
                    change=resize(c.reshape(b*2, 1, *c.shape[-2:]), image_size).reshape(b, 2, *image_size))

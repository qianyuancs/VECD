"""Pool integer counts before computing a metric; do not average image IoUs."""
import numpy as np


def confusion(prediction, target, change=None, valid=None):
    p, y = np.asarray(prediction, bool), np.asarray(target, bool)
    if p.shape != y.shape:
        raise ValueError("Prediction and GT must share native resolution")
    v = np.ones_like(y) if valid is None else np.asarray(valid, bool)
    c = y if change is None else np.asarray(change, bool)
    return np.asarray([(p & y & v).sum(), (p & ~y & v).sum(), (~p & y & v).sum(),
                       (~p & ~y & v).sum(), (p & c & ~y & v).sum(), (c & ~y & v).sum()], np.int64)


def measures(counts):
    tp, fp, fn, tn, df, dp = map(int, counts)
    foreground = tp/max(1, tp+fp+fn)
    background = tn/max(1, tn+fp+fn)
    return dict(counts=list(map(int, counts)), binary_mIoU=.5*(foreground+background),
                foreground_IoU=foreground, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
                F1=2*tp/max(1, 2*tp+fp+fn), distractor_FPR=df/max(1, dp))

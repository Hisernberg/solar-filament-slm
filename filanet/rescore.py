"""Learned component rescoring (a tiny gradient-boosted "instance quality head").

Given out-of-fold probability maps, every candidate component gets
    e = mean over annotators of IoU*1{IoU>0.5}     (its expected contribution to the PQ numerator)
A LightGBM regressor predicts e from cheap component features (confidence, spine evidence,
shape, solar position, contrast). A component is kept when e_hat > tau, tau chosen on OOF data
(PQ-optimal rule derived in postproc.py: keep iff p*q > PQ/2).

The scorer is ~100 KB, CPU-only, and adds < 5 ms per image.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .evaluate import ImageComps, component_targets, pq_select
from .postproc import FEATURES


def table(cache: dict[str, ImageComps]):
    X, y, hit, owner = [], [], [], []
    for f, ic in cache.items():
        if ic.n == 0:
            continue
        e, h = component_targets(ic)
        X.append(np.stack([ic.feats[k] for k in FEATURES], 1)); y.append(e); hit.append(h)
        owner += [f] * ic.n
    return np.concatenate(X), np.concatenate(y), np.concatenate(hit), np.array(owner)


def lgb_params(seed=0):
    return dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed)


def fit_oof(cache: dict[str, ImageComps], fold_of_file: dict[str, int], rounds: int = 400):
    """Grouped (by CV fold) out-of-fold rescorer predictions -> per-file scores; plus tau sweep."""
    import lightgbm as lgb
    X, y, _, owner = table(cache)
    folds = np.array([fold_of_file[o] for o in owner])
    pred = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        b = lgb.train(lgb_params(), lgb.Dataset(X[tr], y[tr]), rounds)
        pred[va] = b.predict(X[va])
    scores = {}
    for f in cache:
        scores[f] = pred[owner == f]
    return scores, pred, y


def sweep_tau(cache, scores, taus=np.linspace(0.05, 0.5, 46)):
    best = None
    rows = []
    for t in taus:
        keep = {f: scores[f] > t if cache[f].n else np.zeros(0, bool) for f in cache}
        r = pq_select(cache, keep)
        rows.append((float(t), r.pq))
        if best is None or r.pq > best[1].pq:
            best = (float(t), r)
    return best, rows


def fit_final(cache: dict[str, ImageComps], out_path: str, tau: float, rounds: int = 400, meta: dict | None = None):
    import lightgbm as lgb
    X, y, _, _ = table(cache)
    b = lgb.train(lgb_params(), lgb.Dataset(X, y), rounds)
    b.save_model(out_path)
    json.dump(dict(tau=tau, features=FEATURES, **(meta or {})), open(os.path.splitext(out_path)[0] + ".json", "w"),
              indent=1)
    return b


class Rescorer:
    def __init__(self, path: str):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=path)
        meta = json.load(open(os.path.splitext(path)[0] + ".json"))
        self.tau, self.features, self.meta = meta["tau"], meta["features"], meta

    def keep(self, feats: dict) -> np.ndarray:
        if len(feats[self.features[0]]) == 0:
            return np.zeros(0, bool)
        X = np.stack([feats[k] for k in self.features], 1)
        return self.booster.predict(X) > self.tau

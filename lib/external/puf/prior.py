# Parts adapted from PUF (https://github.com/yyyyangyi/PUF,
# Merging/spatial_prior.py and Scripts/dataset/compute_relation_prior.py),
# Apache License 2.0, Copyright (c) the PUF authors.  Re-implemented for the
# Action Genome person->object vocabulary; see setup/EXT_PUF.md.
"""Class-conditional relation prior for never-co-observed person-object pairs.

PUF's prior is ``alpha = s * P_exist(c_i, c_j) * normalize(P_class(r|c_i,c_j) * P_spatial(r|x_i,x_j))``.
In AG the subject is always the person, so the class term is indexed by the
object class only:

* ``P_att[c]``   (3,)  categorical attention frequencies (Laplace 0.01);
* ``P_spa[c]``   (6,)  per-predicate Bernoulli rates (spatial is multi-label);
* ``P_con[c]``   (17,) per-predicate Bernoulli rates (contacting is multi-label);
* ``P_exist[c]``        fraction of annotated (person, c) world slots that carry
                        at least one relation label.

``*_unobs`` variants are fitted on the unobserved (``visible == False``) slots
only.  PUF itself has no visibility-conditioned prior; it is used only by the
explicitly-labelled extension arm ``puf_prior_vis``.

Spatial factor (replacing PUF's ScanNet support-relation scores, which have no
AG counterpart): with ``delta = x_obj - x_person`` in the z-up canonical frame
and ``d = |delta| / sigma_d``, ``v = delta_z / sigma_d``, the Bernoulli
predicates get a log-likelihood-ratio shift

    contact predicates (all but not_contacting): 1 - d   (proximity, neutral at d = 1)
    not_contacting:                             d - 1
    above / beneath:                            +v / -v
    in:                                         1 - d
    in_front_of / behind / on_the_side_of:      0        (no facing direction)

and the attention head has no spatial factor.  Without geometry (the object was
never observed) the spatial factor is dropped.
"""
from __future__ import annotations

import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

ATT = ["looking_at", "not_looking_at", "unsure"]
SPA = ["above", "beneath", "in_front_of", "behind", "on_the_side_of", "in"]
CON = ["carrying", "covered_by", "drinking_from", "eating", "have_it_on_the_back", "holding",
       "leaning_on", "lying_on", "not_contacting", "other_relationship", "sitting_on",
       "standing_on", "touching", "twisting", "wearing", "wiping", "writing_on"]
N_OBJ = 37
_NOT_CONTACTING = CON.index("not_contacting")


def _name_to_idx():
    from lib.mllm.data.worldbbox import NAME_TO_IDX
    return NAME_TO_IDX


def _count_file(path):
    n2i = _name_to_idx()
    z = lambda *s: np.zeros(s, dtype=np.float64)  # noqa: E731
    acc = {v: {"att": z(N_OBJ, 3), "att_n": z(N_OBJ), "spa": z(N_OBJ, 6), "con": z(N_OBJ, 17),
               "n_rel": z(N_OBJ), "n_present": z(N_OBJ)} for v in ("all", "unobs")}
    try:
        with open(path, "rb") as f:
            a = pickle.load(f)
    except Exception:
        return acc
    for fr in a.get("frames", {}).values():
        for o in fr.get("object_info_list", []) or []:
            lab = o.get("label") or o.get("class", "")
            c = n2i.get(lab, n2i.get(o.get("class", ""), 0))
            if c <= 1:
                continue
            att = [r for r in (o.get("attention_relationship") or []) if r in ATT]
            spa = [r for r in (o.get("spatial_relationship") or []) if r in SPA]
            con = [r for r in (o.get("contacting_relationship") or []) if r in CON]
            buckets = ["all"] + (["unobs"] if not o.get("visible", True) else [])
            for b in buckets:
                A = acc[b]
                A["n_present"][c] += 1
                if not (att or spa or con):
                    continue
                A["n_rel"][c] += 1
                if att:
                    A["att"][c, ATT.index(att[0])] += 1
                    A["att_n"][c] += 1
                for r in set(spa):
                    A["spa"][c, SPA.index(r)] += 1
                for r in set(con):
                    A["con"][c, CON.index(r)] += 1
    return acc


def fit_prior(train_dir: str, out_path: str, workers: int = 16, smooth: float = 0.01, limit: int = 0):
    files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))
    if limit:
        files = files[:limit]
    tot = None
    with ProcessPoolExecutor(workers) as ex:
        for acc in ex.map(_count_file, files, chunksize=16):
            if tot is None:
                tot = acc
            else:
                for b in tot:
                    for k in tot[b]:
                        tot[b][k] += acc[b][k]
    out = {"n_files": len(files), "train_dir": train_dir}
    for b, suf in (("all", ""), ("unobs", "_unobs")):
        A = tot[b]
        n = np.maximum(A["n_rel"], 1.0)
        out["P_att" + suf] = (A["att"] + smooth) / (A["att_n"][:, None] + 3 * smooth)
        out["P_spa" + suf] = (A["spa"] + smooth) / (n[:, None] + 2 * smooth)
        out["P_con" + suf] = (A["con"] + smooth) / (n[:, None] + 2 * smooth)
        out["P_exist" + suf] = np.where(A["n_present"] > 0, A["n_rel"] / np.maximum(A["n_present"], 1), 0.0)
        out["n_pairs" + suf] = A["n_rel"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
             n_files=np.array(len(files)))
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RelationPrior:
    """Dirichlet / Beta pseudo-counts for a (person, object-class) edge."""

    def __init__(self, path: str, strength: float = 1.0, sigma_d: float = 0.5, use_spatial: bool = True):
        z = np.load(path)
        self.P = {k: z[k] for k in z.files}
        self.strength = strength
        self.sigma_d = sigma_d          # in units of the video scale (median camera depth)
        self.use_spatial = use_spatial

    def probs(self, c: int, delta: Optional[np.ndarray], scale: float, unobs: bool = False):
        """(p_att(3), p_spa(6), p_con(17), p_exist) for object class c."""
        suf = "_unobs" if unobs else ""
        p_att = self.P["P_att" + suf][c].copy()
        p_spa = self.P["P_spa" + suf][c].copy()
        p_con = self.P["P_con" + suf][c].copy()
        p_ex = float(self.P["P_exist" + suf][c])
        if unobs and self.P["n_pairs_unobs"][c] < 20:        # too few samples: fall back
            return self.probs(c, delta, scale, unobs=False)
        if self.use_spatial and delta is not None and np.all(np.isfinite(delta)):
            sd = self.sigma_d * scale
            d = float(np.linalg.norm(delta)) / sd
            v = float(delta[2]) / sd
            prox = 1.0 - d
            ls = np.zeros(6)
            ls[SPA.index("above")] = v
            ls[SPA.index("beneath")] = -v
            ls[SPA.index("in")] = prox
            lc = np.full(17, prox)
            lc[_NOT_CONTACTING] = -prox
            p_spa = _sigmoid(_logit(p_spa) + ls)
            p_con = _sigmoid(_logit(p_con) + lc)
        return p_att, p_spa, p_con, p_ex

    def pseudo_counts(self, c, delta, scale, unobs=False):
        """PUF: alpha = s * P_exist * P  (categorical att; Beta (pos, neg) for multi-label heads)."""
        p_att, p_spa, p_con, p_ex = self.probs(c, delta, scale, unobs)
        w = self.strength * p_ex
        return {"att": w * p_att,
                "spa": np.stack([w * p_spa, w * (1 - p_spa)], -1),
                "con": np.stack([w * p_con, w * (1 - p_con)], -1)}

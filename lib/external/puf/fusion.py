"""Per-video causal graph fusion + per-timestamp read-out (PUF adapted to WSGG).

Input per video: the Track-1 front-end dump records (W-DSGDetr++; one record
per frame, ``frame_t`` order) and the Pi3 geometry cache of
:mod:`lib.external.puf.geometry`.  Output: the same records with the three
relation distributions replaced (every GT field is copied unchanged, so the
scored slots are exactly the front-end's), plus ``pred_corners`` (N_max, 8, 3)
node boxes and a per-pair ``puf_source`` code.

Arms (``FusionConfig.arm``):

  frontend       observed pairs = front-end, unobserved = 0 (no memory at all)
  lks            naive last-known state: unobserved pair = the front-end output
                 the last time that object class was observed (causal)
  lks_bi         as lks but nearest observed frame in either direction
                 (the zero-order hold of baselines/lks_buffer, non-causal)
  fross          PUF - prior - uncertainty: FROSS hard association (same class +
                 Hellinger < 0.85), vote counting (one-hot argmax / >0.5 votes)
  puf            PUF - prior: likelihood association (spatial x JSD), JPDA
                 birth, soft Dirichlet class + relation evidence
  puf_prior      PUF full: + class-conditional prior (co-occurrence x spatial
                 decay x existence gate) completing every edge
  puf_prior_vis  EXTENSION (not in PUF): prior fitted on unobserved slots is used
                 for currently-unobserved objects

Person: a per-frame node (never associated); relation evidence is accumulated
on the (person track, object node) edge.  AG videos have one person.

``puf_source`` codes: 0 observed (fused evidence), 1 memory (node evidence, object
unobserved now), 2 prior only, 3 nothing (zeros), 4 front-end passthrough (lks arms).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from lib.external.puf.puf_core import association_likelihood, hellinger, jpda_marginals, merge_gaussians

N_OBJ = 37
PERSON = 1
SRC_OBS, SRC_MEM, SRC_PRIOR, SRC_NONE, SRC_FE = 0, 1, 2, 3, 4


@dataclass
class FusionConfig:
    arm: str = "puf_prior"
    mode: str = "predcls"
    # PUF association (README 3DSSG command: lambda_birth 0.4, sigma_jsd 0.3)
    lambda_birth: float = 0.4
    tau_birth: float = 0.5
    beta_min: float = 0.05
    sigma_jsd: float = 0.3
    l2_gate: float = 1.5            # x video scale (median camera depth); PUF: 3 m
    obs_strength: float = 1.0
    obj_obs_strength: float = 1.0
    # FROSS hard association
    hellinger_threshold: float = 0.85
    # dynamic-person edge forgetting (1.0 = PUF static accumulation)
    edge_decay: float = 1.0
    # prior
    completion_threshold: float = 0.0
    max_node_pts: int = 1500
    emit_boxes: bool = True


class _Node:
    __slots__ = ("cls", "mean", "cov", "n", "pts", "att", "spa", "con", "n_edge", "last_t", "box", "box_dirty")

    def __init__(self, cls_alpha, geom, t):
        self.cls = cls_alpha.astype(np.float64).copy()
        if geom is not None:
            self.mean = geom[0].astype(np.float64)
            self.cov = geom[1].astype(np.float64)
            self.n = geom[2]
            self.pts = geom[3].astype(np.float32)
        else:
            self.mean = self.cov = self.pts = None
            self.n = 0
        self.att = np.zeros(3)
        self.spa = np.zeros((6, 2))
        self.con = np.zeros((17, 2))
        self.n_edge = 0.0
        self.last_t = t
        self.box = None
        self.box_dirty = True

    @property
    def label(self):
        return int(np.argmax(self.cls))

    @property
    def has_geom(self):
        return self.mean is not None

    def add_evidence(self, ev, w=1.0):
        if ev is None:
            return
        self.att += w * ev["att"]
        self.spa += w * ev["spa"]
        self.con += w * ev["con"]
        self.n_edge += w

    def merge_geom(self, geom, max_pts, rng):
        if geom is None:
            return
        if self.mean is None:
            self.mean, self.cov, self.n = geom[0].astype(np.float64), geom[1].astype(np.float64), geom[2]
            self.pts = geom[3].astype(np.float32)
        else:
            self.mean, self.cov = merge_gaussians(geom[0].astype(np.float64), geom[1].astype(np.float64), geom[2],
                                                  self.mean, self.cov, self.n)
            self.n += geom[2]
            self.pts = np.concatenate([self.pts, geom[3].astype(np.float32)])
            if len(self.pts) > max_pts:
                self.pts = self.pts[rng.choice(len(self.pts), max_pts, replace=False)]
        self.box_dirty = True

    def obb(self):
        if self.pts is None or len(self.pts) < 4:
            return None
        if self.box_dirty or self.box is None:
            from lib.mllm.data.geometry import obb_floor_parallel_from_points
            p = self.pts.astype(np.float64)
            lo, hi = np.percentile(p, 5, axis=0), np.percentile(p, 95, axis=0)
            q = p[np.all((p >= lo) & (p <= hi), axis=1)]
            if len(q) < 4:
                q = p
            self.box = obb_floor_parallel_from_points(q).astype(np.float32)
            self.box_dirty = False
        return self.box


def _class_alpha(c: int, score: float, mode: str, strength: float) -> np.ndarray:
    a = np.zeros(N_OBJ)
    if mode == "predcls" or score >= 1.0:
        a[c] = 1.0                                   # PUF "GT mode": one-hot
    else:
        s = float(np.clip(score, 1e-3, 1.0))
        a[2:] = (1.0 - s) / (N_OBJ - 3)              # detector confidence spread over the other object classes
        a[c] = s
    return strength * a


def _evidence(att, spa, con, hard: bool):
    if hard:   # FROSS vote counting
        a = np.zeros(3)
        a[int(np.argmax(att))] = 1.0
        sp = (spa > 0.5).astype(np.float64)
        cp = (con > 0.5).astype(np.float64)
        return {"att": a, "spa": np.stack([sp, 1 - sp], -1), "con": np.stack([cp, 1 - cp], -1)}
    att = np.clip(att.astype(np.float64), 0, None)
    s = att.sum()
    att = att / s if s > 0 else np.full(3, 1 / 3)
    sp = np.clip(spa.astype(np.float64), 0, 1)
    cp = np.clip(con.astype(np.float64), 0, 1)
    return {"att": att, "spa": np.stack([sp, 1 - sp], -1), "con": np.stack([cp, 1 - cp], -1)}


def _readout(att, spa, con):
    sa = att.sum()
    a = att / sa if sa > 0 else np.zeros(3)
    s = spa[:, 0] / np.maximum(spa.sum(-1), 1e-12)
    c = con[:, 0] / np.maximum(con.sum(-1), 1e-12)
    return a.astype(np.float32), s.astype(np.float32), c.astype(np.float32)


class VideoFusion:
    def __init__(self, cfg: FusionConfig, scale: float, prior=None):
        self.cfg = cfg
        self.scale = max(float(scale), 1e-6)
        self.prior = prior
        self.nodes: List[_Node] = []
        self.rng = np.random.RandomState(0)
        self.stats = {"obs": 0, "obs_nogeom": 0, "births": 0, "assoc": 0}

    # ---------------- association ----------------
    def _associate_fross(self, c, geom, cand_ids):
        same = [j for j in cand_ids if self.nodes[j].label == c]
        if not same:
            return None
        if geom is None:
            return max(same, key=lambda j: (self.nodes[j].cls.sum(), self.nodes[j].last_t))
        g_ids = [j for j in same if self.nodes[j].has_geom]
        if not g_ids:
            return same[0]
        h = hellinger(geom[0].astype(np.float64), geom[1].astype(np.float64),
                      np.stack([self.nodes[j].mean for j in g_ids]), np.stack([self.nodes[j].cov for j in g_ids]))
        k = int(np.argmin(h))
        return g_ids[k] if h[k] < self.cfg.hellinger_threshold else None

    def _associate_puf(self, alpha, geom, cand_ids):
        """-> (argmax node or None for birth, [(node, beta)] soft weights)."""
        cfg = self.cfg
        if geom is not None:
            gated = []
            for j in cand_ids:
                n = self.nodes[j]
                if not n.has_geom or np.linalg.norm(n.mean - geom[0]) < cfg.l2_gate * self.scale:
                    gated.append(j)
        else:
            gated = list(cand_ids)
        if not gated:
            return None, []
        nodes = [self.nodes[j] for j in gated]
        has_g = np.array([n.has_geom for n in nodes])
        means = np.stack([n.mean if n.has_geom else np.zeros(3) for n in nodes])
        covs = np.stack([n.cov if n.has_geom else np.eye(3) for n in nodes])
        L = association_likelihood(None if geom is None else geom[0].astype(np.float64),
                                   None if geom is None else geom[1].astype(np.float64),
                                   alpha, means, covs, np.stack([n.cls for n in nodes]),
                                   cfg.sigma_jsd, has_geom=has_g)
        beta, beta_birth = jpda_marginals(L, cfg.lambda_birth)
        if beta_birth > cfg.tau_birth:
            return None, []
        best = gated[int(np.argmax(beta))]
        soft = [(gated[i], float(beta[i])) for i in range(len(gated)) if beta[i] >= cfg.beta_min]
        return best, soft

    # ---------------- one frame ----------------
    def step(self, t, observations, person_geom):
        """observations: list of dicts {slot, cls, score, geom, ev}.  Returns slot -> node id."""
        cfg = self.cfg
        if cfg.edge_decay < 1.0:
            for n in self.nodes:
                n.att *= cfg.edge_decay
                n.spa *= cfg.edge_decay
                n.con *= cfg.edge_decay
                n.n_edge *= cfg.edge_decay
        prev_ids = list(range(len(self.nodes)))       # PUF: associate against pre-frame nodes only
        assoc = {}
        hard = cfg.arm == "fross"
        for ob in observations:
            self.stats["obs"] += 1
            if ob["geom"] is None:
                self.stats["obs_nogeom"] += 1
            alpha = _class_alpha(ob["cls"], ob["score"], cfg.mode, cfg.obj_obs_strength)
            ev = None if ob["ev"] is None else _evidence(*ob["ev"], hard=hard)
            if hard:
                j = self._associate_fross(ob["cls"], ob["geom"], prev_ids)
                soft = [(j, 1.0)] if j is not None else []
                if j is not None:
                    self.nodes[j].cls[ob["cls"]] += 1.0
            else:
                j, soft = self._associate_puf(alpha, ob["geom"], prev_ids)
                for jj, w in soft:
                    self.nodes[jj].cls += w * alpha
            if j is None:
                n = _Node(alpha if not hard else np.eye(N_OBJ)[ob["cls"]], ob["geom"], t)
                n.add_evidence(ev, cfg.obs_strength)
                self.nodes.append(n)
                j = len(self.nodes) - 1
                self.stats["births"] += 1
            else:
                self.stats["assoc"] += 1
                self.nodes[j].merge_geom(ob["geom"], cfg.max_node_pts, self.rng)
                for jj, w in soft:
                    self.nodes[jj].add_evidence(ev, w * cfg.obs_strength)
                self.nodes[j].last_t = t
            assoc[ob["slot"]] = j
        return assoc

    def node_for_class(self, c) -> Optional[int]:
        """Read-out slot -> node map for a slot that is not observed now: the node
        carrying class-c evidence, preferring nodes whose argmax is c (or that hold
        at least half an observation of c), then nodes with relation evidence,
        then the most class-c mass.  Under PUF's soft class updates a node's argmax
        can drift to a neighbour's class, so argmax == c is not required.  The 0.1
        floor excludes the confidence mass an sgdet detection spreads over the
        other classes ((1 - score) / 34 < 0.03)."""
        best, key = None, None
        for j, n in enumerate(self.nodes):
            if n.cls[c] < 0.1:
                continue
            k = (n.label == c or n.cls[c] >= 0.5, n.n_edge > 0, n.cls[c], n.last_t)
            if key is None or k > key:
                best, key = j, k
        return best

    def edge_output(self, j, c, person_geom, unobserved):
        """Relation distributions for (person, node j | class c).  -> (att, spa, con, src)."""
        cfg = self.cfg
        use_prior = cfg.arm in ("puf_prior", "puf_prior_vis") and self.prior is not None
        n = self.nodes[j] if j is not None else None
        att = np.zeros(3)
        spa = np.zeros((6, 2))
        con = np.zeros((17, 2))
        src = SRC_NONE
        if n is not None and n.n_edge > 0:
            att, spa, con = n.att.copy(), n.spa.copy(), n.con.copy()
            src = SRC_MEM if unobserved else SRC_OBS
        if use_prior:
            delta = None
            if n is not None and n.has_geom and person_geom is not None:
                delta = n.mean - person_geom[0].astype(np.float64)
            pc = self.prior.pseudo_counts(c, delta, self.scale,
                                          unobs=(cfg.arm == "puf_prior_vis" and unobserved))
            if src == SRC_NONE:
                p_att = pc["att"] / max(pc["att"].sum(), 1e-12)
                if p_att.max() <= cfg.completion_threshold:
                    return _readout(att, spa, con) + (SRC_NONE,)
                src = SRC_PRIOR
            att, spa, con = att + pc["att"], spa + pc["spa"], con + pc["con"]
        return _readout(att, spa, con) + (src,)


def _observations_for_frame(rec, g, cfg: FusionConfig):
    pv = rec["pair_valid"].astype(bool)
    vis = rec["visibility_mask"].astype(bool)
    valid = rec["valid_mask"].astype(bool)
    cls = rec["object_classes"]
    first_pair = {}
    for k in np.nonzero(pv)[0]:
        o = int(rec["object_idx"][k])
        first_pair.setdefault(o, k)
    obs = []
    for i in np.nonzero(valid & vis)[0]:
        c = int(cls[i])
        if c == PERSON or c <= 0:
            continue
        k = first_pair.get(int(i))
        ev = None
        if k is not None and vis[int(rec["person_idx"][k])]:
            ev = (rec["attention_distribution"][k], rec["spatial_distribution"][k],
                  rec["contacting_distribution"][k])
        score = float(g["scores"][i]) if (g is not None and i < len(g["scores"])) else 1.0
        geom = g["obs"].get(int(i)) if g is not None else None
        obs.append({"slot": int(i), "cls": c, "score": score, "geom": geom, "ev": ev})
    return obs


def _person_geom(rec, g):
    if g is None:
        return None
    pv = rec["pair_valid"].astype(bool)
    ps = rec["person_idx"][pv]
    p = int(ps[0]) if len(ps) else 0
    return g["obs"].get(p)


def run_video_graph(records: List[dict], geom_mode: Optional[dict], cfg: FusionConfig, scale: float,
                    prior=None) -> (List[dict], dict):
    """Graph arms (fross / puf / puf_prior / puf_prior_vis)."""
    vf = VideoFusion(cfg, scale, prior)
    per_frame = geom_mode["per_frame"] if geom_mode is not None else None
    out = []
    for t, rec in enumerate(records):
        g = per_frame[t] if (per_frame is not None and t < len(per_frame)) else None
        pg = _person_geom(rec, g)
        assoc = vf.step(t, _observations_for_frame(rec, g, cfg), pg)
        new = dict(rec)
        K = len(rec["pair_valid"])
        att = np.zeros_like(rec["attention_distribution"])
        spa = np.zeros_like(rec["spatial_distribution"])
        con = np.zeros_like(rec["contacting_distribution"])
        src = np.full(K, -1, dtype=np.int8)
        vis = rec["visibility_mask"].astype(bool)
        cache = {}
        for k in np.nonzero(rec["pair_valid"].astype(bool))[0]:
            o = int(rec["object_idx"][k])
            if o not in cache:
                c = int(rec["object_classes"][o])
                if vis[o] and o in assoc:
                    j, unobs = assoc[o], False
                else:
                    j, unobs = vf.node_for_class(c), True
                cache[o] = vf.edge_output(j, c, pg, unobs) + (j,)
            a, s, cc, sr, _ = cache[o]
            att[k], spa[k], con[k], src[k] = a, s, cc, sr
        new["attention_distribution"] = att
        new["spatial_distribution"] = spa
        new["contacting_distribution"] = con
        new["puf_source"] = src
        if cfg.emit_boxes:
            N = len(rec["object_classes"])
            corners = np.full((N, 8, 3), np.nan, dtype=np.float32)
            for o, v in cache.items():
                j = v[-1]
                if j is not None:
                    b = vf.nodes[j].obb()
                    if b is not None:
                        corners[o] = b
            new["pred_corners"] = corners
        out.append(new)
    st = dict(vf.stats)
    st["nodes"] = len(vf.nodes)
    return out, st


def run_video_lks(records: List[dict], cfg: FusionConfig) -> (List[dict], dict):
    """frontend / lks / lks_bi arms: identity = object class (as in the AG pair keys)."""
    T = len(records)
    obs_at: Dict[int, List[int]] = {}
    seen: Dict[int, Dict[int, tuple]] = {}
    for t, rec in enumerate(records):
        vis = rec["visibility_mask"].astype(bool)
        seen[t] = {}
        for k in np.nonzero(rec["pair_valid"].astype(bool))[0]:
            o = int(rec["object_idx"][k])
            c = int(rec["object_classes"][o])
            if vis[o] and c not in seen[t]:
                seen[t][c] = (rec["attention_distribution"][k], rec["spatial_distribution"][k],
                              rec["contacting_distribution"][k])
                obs_at.setdefault(c, []).append(t)
    out = []
    for t, rec in enumerate(records):
        new = dict(rec)
        vis = rec["visibility_mask"].astype(bool)
        att = rec["attention_distribution"].copy()
        spa = rec["spatial_distribution"].copy()
        con = rec["contacting_distribution"].copy()
        K = len(rec["pair_valid"])
        src = np.full(K, -1, dtype=np.int8)
        for k in np.nonzero(rec["pair_valid"].astype(bool))[0]:
            o = int(rec["object_idx"][k])
            if vis[o]:
                src[k] = SRC_FE
                continue
            c = int(rec["object_classes"][o])
            ts = obs_at.get(c, [])
            cand = None
            if cfg.arm == "lks":
                prev = [u for u in ts if u < t]
                cand = prev[-1] if prev else None
            elif cfg.arm == "lks_bi":
                if ts:
                    cand = min(ts, key=lambda u: (abs(u - t), u > t))
            if cand is None:
                att[k], spa[k], con[k] = 0, 0, 0
                src[k] = SRC_NONE
            else:
                att[k], spa[k], con[k] = seen[cand][c]
                src[k] = SRC_MEM
        new["attention_distribution"] = att
        new["spatial_distribution"] = spa
        new["contacting_distribution"] = con
        new["puf_source"] = src
        out.append(new)
    return out, {}

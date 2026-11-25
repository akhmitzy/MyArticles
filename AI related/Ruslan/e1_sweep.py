"""E1: exact (population) fixed points on random MDPs.
(a) sweep transition stochasticity, next-state sampling only and next-state+next-action sampling
(b) sweep reward scale (label-form bias is not scale invariant)."""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from common import *

g = 0.9
seeds = range(20)


def evaluate(P, R, beta, pi, lam, sample_a):
    ps = np.ones(P.shape[0]) / P.shape[0]
    pred = predicted(P, R, beta, pi, g, lam)
    Qb = q_eval(P, R, beta, g)
    _, j1 = actor_obj_onestep(Qb, beta, pi, ps, lam)
    res = {}
    for form in ["scaled", "label"]:
        Q = fixed_point(P, R, beta, pi, g, lam, form, sample_a=sample_a)
        res[form + "_q"] = float(np.max(np.abs(Q - pred) / pred))
        res[form + "_under"] = bool(np.all(Q <= pred * (1 + 1e-9)))
        _, jc = actor_obj_reg(Q, pi, ps)
        res[form + "_j"] = float(np.max(np.abs(jc - j1)))
    return res


def agg(lst):
    keys = lst[0].keys(); out = {}
    for k in keys:
        v = np.array([d[k] for d in lst], dtype=float)
        out[k] = [float(v.mean()), float(v.std()), float(v.max()), float(v.min())]
    return out


out = {"stoch": [], "scale": []}
concs = [None, 30.0, 3.0, 1.0, 0.3, 0.1, 0.03]
for lam in [1.0, 0.5]:
    for sample_a in [False, True]:
        for conc in concs:
            lst, Hs = [], []
            for sd in seeds:
                P, R, beta, pi = random_mdp(sd, conc=conc if conc else 1.0, det=conc is None)
                Hs.append(-(P * np.log(np.where(P > 0, P, 1))).sum(2).mean())
                lst.append(evaluate(P, R, beta, pi, lam, sample_a))
            a = agg(lst); a.update(lam=lam, sample_a=sample_a, conc=conc, H=float(np.mean(Hs)))
            out["stoch"].append(a)
            print(f"lam={lam} sa={sample_a} conc={conc} H={a['H']:.2f} | scaled max q-err {a['scaled_q'][2]:.1e} "
                  f"j {a['scaled_j'][2]:.1e} | label q-err {a['label_q'][0]:.3f}±{a['label_q'][1]:.3f} "
                  f"j {a['label_j'][0]:.3f} under={a['label_under'][3]}")

for sample_a in [False, True]:
    for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
        lst = []
        for sd in seeds:
            P, R, beta, pi = random_mdp(sd, conc=0.1)
            lst.append(evaluate(P, R * scale, beta, pi, 1.0, sample_a))
        a = agg(lst); a.update(scale=scale, sample_a=sample_a)
        out["scale"].append(a)
        print(f"scale={scale} sa={sample_a} | scaled {a['scaled_q'][2]:.1e} | label q-err {a['label_q'][0]:.3f} j {a['label_j'][0]:.3f}")
json.dump(out, open("e1.json", "w"), indent=1)

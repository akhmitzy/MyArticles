"""E3: single-sample stochastic-gradient training of the regularised critic on an offline dataset.

Everything is sampled: (s,a,s') from the dataset, a' ~ pi(.|s'), negatives a- ~ pi(.|s), and for the
label form the TD target at the negative pair uses a dataset successor of (s,a-).
Target network is refreshed every K steps (fitted Q-iteration). We compare three losses:
  ce-scaled  : -y log D(Q) at positives, -log(1-D(Q)) at negatives     (Bregman/CE member)
  lsif-scaled: -y Q at positives, Q^2/2 at negatives                    (squared/LSIF member)
  ce-label   : -(y/(1+y)) log D at positives, -(1/(1+y)) log(1-D) at negatives
Error is measured against the theorem's prediction Q^beta * beta/pi on the empirical MDP."""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from common import *

g, n, slip, goal_r = 0.9, 4, 0.2, 1.0
S, A = n * n, 4
rng0 = np.random.default_rng(123)
beta_true = 0.9 * rng0.dirichlet(np.ones(4), size=S) + 0.025
P, R = slippery_grid(n, slip, goal_r=goal_r)
data = collect_dataset(P, beta_true, 300, 25, seed=7)
Ph, bh, ps, nsa = empirical(data, S, A)
pi = np.random.default_rng(5).dirichlet(np.ones(A) * 2, size=S) * 0.5 + 0.5 * bh   # a fixed target policy
pred = predicted(Ph, R, bh, pi, g, 1.0)
# successor lists per (s,a) for label-form negatives
succ = {(s, a): data[(data[:, 0] == s) & (data[:, 1] == a), 2] for s in range(S) for a in range(A)}
maxn = max(len(v) for v in succ.values())
SUCC = np.zeros((S, A, maxn), dtype=int); NS = np.zeros((S, A), dtype=int)
for (s_, a_), v in succ.items():
    NS[s_, a_] = len(v); SUCC[s_, a_, :len(v)] = v
label_fp = fixed_point(Ph, R, bh, pi, g, 1.0, "label", sample_a=True)
cdf_pi = np.cumsum(pi, 1)
sig = lambda u: 1 / (1 + np.exp(-u))


def sample_actions(rng, states):
    u = rng.random(len(states))
    return (u[:, None] > cdf_pi[states]).sum(1).clip(0, A - 1)


def run(form, seed, outer=400, K=400, B=256, lr=0.3):
    rng = np.random.default_rng(seed)
    if form.startswith("ce"):
        par = np.zeros((S, A))           # u = log Q
        getQ = lambda p: np.exp(p)
    else:
        par = np.ones((S, A))            # Q directly
        getQ = lambda p: p
    errs = []
    step = 0
    tail = []
    for k in range(outer):
        Qt = getQ(par).copy()
        for i in range(K):
            step += 1
            eta = lr / (1 + step / 5000) ** 0.6
            idx = rng.integers(len(data), size=B)
            s, a, s2 = data[idx, 0], data[idx, 1], data[idx, 2]
            a2 = sample_actions(rng, s2)
            y = R[s, a] + g * Qt[s2, a2]
            an = sample_actions(rng, s)
            G = np.zeros((S, A))
            if form == "ce-scaled":
                np.add.at(G, (s, a), -y * (1 - sig(par[s, a])))
                np.add.at(G, (s, an), sig(par[s, an]))
            elif form == "lsif-scaled":
                np.add.at(G, (s, a), -y)
                np.add.at(G, (s, an), par[s, an])
            elif form == "ce-label":
                np.add.at(G, (s, a), -(y / (1 + y)) * (1 - sig(par[s, a])))
                # target at the negative pair: sample a dataset successor of (s, an)
                s3 = SUCC[s, an, (rng.random(B) * NS[s, an]).astype(int)]
                yn = R[s, an] + g * Qt[s3, sample_actions(rng, s3)]
                np.add.at(G, (s, an), (1 / (1 + yn)) * sig(par[s, an]))
            # per-entry normalisation by visit frequency keeps step sizes comparable across (s,a)
            freq = np.maximum(ps[:, None] * np.maximum(bh, pi), 1e-3)
            par -= eta * G / (B * freq)
        Q = getQ(par)
        if k >= outer // 2:
            tail.append(Q.copy())
        errs.append([float(np.median(np.abs(Q - pred) / pred)), float(np.median(np.abs(Q - label_fp) / label_fp))])
    Qavg = np.mean(tail, 0)
    return errs, Qavg


out = {"label_fp_err": float(np.median(np.abs(label_fp - pred) / pred))}
print("population label-form fixed point: median rel err vs theory", out["label_fp_err"])
for form in ["ce-scaled", "lsif-scaled", "ce-label"]:
    curves = []
    for seed in range(3):
        e, Q = run(form, seed)
        curves.append(e)
        avg_err = [float(np.median(np.abs(Q - pred) / pred)), float(np.median(np.abs(Q - label_fp) / label_fp))]
        out.setdefault(form + "_avg", []).append(avg_err)
        print(f"   tail-averaged: vs theory {avg_err[0]:.4f} | vs label fixed pt {avg_err[1]:.4f}")
        print(f"{form} seed {seed}: med rel err vs theory {e[-1][0]:.4f} | vs label-form fixed pt {e[-1][1]:.4f}", flush=True)
    out[form] = curves
json.dump(out, open("e3.json", "w"))

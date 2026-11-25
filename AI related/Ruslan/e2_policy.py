"""E2: policy-level test on an offline dataset from a slippery gridworld.

For each slip probability we build the empirical MDP from a fixed dataset, then MAXIMISE the
critic-regularised actor objective J_form(pi) = E_{p(s)pi(a|s)} log Q_form^pi(s,a) numerically
(softmax policy, L-BFGS, finite differences), where Q_form^pi is the exact fixed point of the
regularised classifier critic with single-sample targets. We compare the maximiser with the
closed-form one-step policy pi*(a|s) ∝ beta(a|s) Q^beta(s,a)."""
import json, sys, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from common import *

g = 0.9
n = 4
out = []
rng0 = np.random.default_rng(123)
beta_true = rng0.dirichlet(np.ones(4) * 1.0, size=n * n)
beta_true = 0.9 * beta_true + 0.1 / 4  # full support


def softmax_pol(theta, mask):
    z = np.where(mask, theta, -np.inf)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


for goal_r in [1.0, 10.0]:
  for slip in [0.0, 0.1, 0.2, 0.4]:
      P, R = slippery_grid(n, slip, goal_r=goal_r)
      data = collect_dataset(P, beta_true, n_traj=300, T=25, seed=7)
      Ph, bh, ps, nsa = empirical(data, n * n, 4)
      assert (ps > 0).all()
      mask = bh > 0
      Qb = q_eval(Ph, R, bh, g)
      pstar = bh * Qb; pstar /= pstar.sum(1, keepdims=True)
      rec = {"slip": slip, "goal_r": goal_r, "N": len(data)}
      for form, sa in [("scaled", True), ("label", False), ("label", True)]:
          def negJ(th):
              pi = softmax_pol(th.reshape(n * n, 4), mask)
              Q = fixed_point(Ph, R, bh, pi, g, 1.0, form, iters=4000, tol=1e-11, sample_a=sa)
              J, _ = actor_obj_reg(Q, pi, ps)
              return -J
          th0 = np.where(mask, np.log(np.maximum(bh, 1e-12)), 0.0).ravel()  # start at beta
          t = time.time()
          res = minimize(negJ, th0, method="L-BFGS-B", options={"maxiter": 500, "eps": 1e-7})
          pi_hat = softmax_pol(res.x.reshape(n * n, 4), mask)
          x, y = pstar[mask], pi_hat[mask]
          r2 = 1 - ((y - x) ** 2).sum() / ((x - x.mean()) ** 2).sum()
          # true-environment discounted return from uniform start
          def ret(pol):
              return float((q_eval(P, R, pol, g) * pol).sum(1).mean())
          key = f"{form}_{'sa' if sa else 's'}"
          rec[key] = {"maxdiff": float(np.abs(x - y).max()), "r2": float(r2), "ret": ret(pi_hat),
                      "pi": pi_hat.tolist(), "nit": int(res.nit)}
          print(f"R={goal_r} slip={slip} {key:9s} max|pi-pi*|={np.abs(x-y).max():.4f} R2={r2:.5f} "
                f"ret={ret(pi_hat):.4f} (pi* ret {ret(pstar):.4f}, beta ret {ret(bh):.4f}) nit={res.nit} {time.time()-t:.0f}s")
      rec["pstar"] = pstar.tolist(); rec["ret_pstar"] = float((q_eval(P, R, pstar, g) * pstar).sum(1).mean())
      rec["ret_beta"] = float((q_eval(P, R, bh, g) * bh).sum(1).mean())
      out.append(rec)
json.dump(out, open("e2.json", "w"), indent=1)

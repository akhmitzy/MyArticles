"""Shared tabular machinery for the stochastic-dynamics equivalence experiments.

Notation follows the paper draft:
  beta  : behavioural policy, shape (S,A)
  m     : negative-sampling / TD mixture policy (1-lam)*beta + lam*pi
  P     : transition kernel (S,A,S)
  R     : positive rewards (S,A)
Targets use the expected next action (expected-SARSA style), so the only
randomness in a TD target comes from the next STATE s' ~ P(.|s,a).
"""
import numpy as np


def q_eval(P, R, pol, g):
    """Exact Q^pol by a linear solve."""
    S, A = R.shape
    M = np.einsum("sat,tb->satb", P, pol).reshape(S * A, S * A)
    return np.linalg.solve(np.eye(S * A) - g * M, R.ravel()).reshape(S, A)


def mixture(beta, pi, lam):
    return (1 - lam) * beta + lam * pi


def fixed_point(P, R, beta, pi, g, lam=1.0, form="scaled", iters=3000, tol=1e-12, sample_a=False):
    if sample_a:
        return _fixed_point_sa(P, R, beta, pi, g, lam, form, iters, tol)
    return _fixed_point_s(P, R, beta, pi, g, lam, form, iters, tol)


def _fixed_point_sa(P, R, beta, pi, g, lam, form, iters, tol):
    """Same as below, but the next ACTION is also sampled: Y = r + g Q(s',a'), (s',a') ~ P x m."""
    m = mixture(beta, pi, lam)
    ratio = np.where(beta > 0, beta / np.maximum(m, 1e-300), 0.0)
    W = P[:, :, :, None] * m[None, None, :, :]
    Q = np.zeros_like(R)
    for _ in range(iters):
        Y = R[:, :, None, None] + g * Q[None, None, :, :]
        T = (W * Y).sum((2, 3)) if form == "scaled" else 1.0 / (W / (1.0 + Y)).sum((2, 3)) - 1.0
        Qn = ratio * T
        if np.max(np.abs(Qn - Q)) < tol:
            return Qn
        Q = Qn
    return Q


def _fixed_point_s(P, R, beta, pi, g, lam=1.0, form="scaled", iters=3000, tol=1e-12):
    """Fixed point of the regularised classifier critic with SAMPLED next states.

    form = "scaled": population minimiser of the target-linear loss,
        Q(s,a) = beta/m * E_{s'}[Y],              Y = r + g * sum_a' m(a'|s') Q(s',a')
    form = "label" : population minimiser of the per-example normalised (probability-label) loss,
        Q(s,a) = beta/m * (1/E_{s'}[1/(1+Y)] - 1)
    Returns Q on the support of beta (entries with beta=0 are 0).
    """
    m = mixture(beta, pi, lam)
    ratio = np.where(beta > 0, beta / np.maximum(m, 1e-300), 0.0)
    Q = np.zeros_like(R)
    for _ in range(iters):
        V = (m * Q).sum(1)                      # V(s') = E_{a'~m} Q(s',a')
        Y = R[:, :, None] + g * V[None, None, :]  # (S,A,S')
        if form == "scaled":
            T = (P * Y).sum(2)
        elif form == "label":
            T = 1.0 / (P * (1.0 / (1.0 + Y))).sum(2) - 1.0
        else:
            raise ValueError(form)
        Qn = ratio * T
        if np.max(np.abs(Qn - Q)) < tol:
            Q = Qn
            break
        Q = Qn
    return Q


def predicted(P, R, beta, pi, g, lam=1.0):
    """Theorem prediction: Q^beta * beta / m."""
    m = mixture(beta, pi, lam)
    Qb = q_eval(P, R, beta, g)
    return np.where(beta > 0, Qb * beta / np.maximum(m, 1e-300), 0.0)


def actor_obj_reg(Q, pi, ps):
    """J_crit(pi) = E_{p(s) pi(a|s)} log Q(s,a)  (restricted to supp pi)."""
    with np.errstate(divide="ignore"):
        lq = np.where(pi > 0, np.log(np.maximum(Q, 1e-300)), 0.0)
    return (ps * (pi * lq).sum(1)).sum(), (pi * lq).sum(1)


def actor_obj_onestep(Qb, beta, pi, ps, lam=1.0):
    """J_1step(pi) = E[log Q^beta + log beta - log m]."""
    m = mixture(beta, pi, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(pi > 0, np.log(Qb) + np.log(beta) - np.log(m), 0.0)
    per_s = (pi * term).sum(1)
    return (ps * per_s).sum(), per_s


# ---------------------------------------------------------------- environments
def random_mdp(seed, S=10, A=4, conc=0.3, det=False):
    rng = np.random.default_rng(seed)
    if det:
        P = np.zeros((S, A, S))
        nxt = rng.integers(0, S, size=(S, A))
        P[np.arange(S)[:, None], np.arange(A)[None, :], nxt] = 1.0
    else:
        P = rng.dirichlet(np.ones(S) * conc, size=(S, A))
    R = rng.uniform(0.1, 1.0, size=(S, A))
    beta = rng.dirichlet(np.ones(A) * 2.0, size=S)
    pi = rng.dirichlet(np.ones(A) * 2.0, size=S)
    return P, R, beta, pi


def slippery_grid(n=4, slip=0.2, goal_r=1.0, base_r=0.05):
    """n x n grid, actions U,D,L,R. With prob 1-slip the intended move, else a uniformly random move.
    Reward depends on (s,a) via expected arrival at goal (positive everywhere)."""
    S, A = n * n, 4
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def step(s, d):
        r, c = divmod(s, n)
        r2, c2 = min(max(r + moves[d][0], 0), n - 1), min(max(c + moves[d][1], 0), n - 1)
        return r2 * n + c2

    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            P[s, a, step(s, a)] += 1 - slip
            for d in range(A):
                P[s, a, step(s, d)] += slip / A
    goal = S - 1
    R = base_r + goal_r * P[:, :, goal]
    return P, R


def collect_dataset(P, beta_true, n_traj, T, seed, s0=None):
    rng = np.random.default_rng(seed)
    S, A, _ = P.shape
    data = []
    for _ in range(n_traj):
        s = rng.integers(S) if s0 is None else s0
        for _ in range(T):
            a = rng.choice(A, p=beta_true[s])
            s2 = rng.choice(S, p=P[s, a])
            data.append((s, a, s2))
            s = s2
    return np.array(data)


def empirical(data, S, A):
    cnt = np.zeros((S, A, S))
    np.add.at(cnt, (data[:, 0], data[:, 1], data[:, 2]), 1)
    nsa = cnt.sum(2)
    Phat = np.where(nsa[..., None] > 0, cnt / np.maximum(nsa[..., None], 1), 1.0 / S)
    ns = nsa.sum(1)
    beta_hat = np.where(ns[:, None] > 0, nsa / np.maximum(ns[:, None], 1), 1.0 / A)
    ps = ns / ns.sum()
    return Phat, beta_hat, ps, nsa

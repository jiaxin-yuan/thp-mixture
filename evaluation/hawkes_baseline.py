#!/usr/bin/env python3
"""Traditional (parametric) Hawkes-process baselines for the UQ4PPM comparison.

Two classical exponential-kernel Hawkes models, fit by maximum likelihood, are
added as non-neural baselines alongside THP / SuTraN / ED-LSTM:

  * Hawkes (uni)    — univariate exp-kernel Hawkes on event *times* only.
                      Comparable to THP-Baseline (time, no activity).
                      λ(t) = μ + Σ_{t_i<t} α·β·e^{-β(t-t_i)}

  * Hawkes (marked) — D-dimensional (one dim per activity) exp-kernel Hawkes.
                      Predicts the next activity *and* its time.
                      Comparable to THP-Mixture.
                      λ_k(t) = μ_k + Σ_j Σ_{t_i^j<t} A_{kj}·β·e^{-β(t-t_i^j)}

Everything is reproduced from the strict-split CSVs (no torch), matching the THP
target conventions in preprocess/dataset.py (all times in DAYS):
  next-event Δt = consecutive timestamp diff within a case;
  remaining_time = (CaseEndTime - Timestamp) / 86400;
  activity vocabulary built from the full_train split (unseen test act → UNK).
Eval is per-position next-event over each test sequence (prefixes [2, n-1]).

Predictions from the fitted intensity:
  next-event time  E[Δt] = ∫₀^∞ exp(-Λ(s)) ds   (closed-form series, see e_gap)
  next activity    argmax_k λ_k at the predicted next time (marked only)
  remaining time   ORACLE-LENGTH rollout: from each prefix, greedily sum E[Δt]
                   for the known number of remaining events (deterministic;
                   Hawkes has no native notion of case end — labeled as such).

Backends (--backend):
  mle  — self-coded numpy/scipy MLE (default-robust; handles tied timestamps,
         which are common in these logs and break `tick`).
  tick — use the `tick` library if importable and viable.
  auto — try tick, fall back to mle on any failure (incl. tied-timestamp errors).

Outputs (so the existing aggregators pick them up):
  results_hawkes_v2/fold0_variation0_<DS>_hawkes.txt
  results_hawkes_marked_v2/fold0_variation0_<DS>_hawkes_marked.txt
  paper_results/per_dataset/<DS>/hawkes_results.json
  logs_hawkes_v2_<DS>.log                                   (fit wall-time)

Usage (env `hawkes`):
  python hawkes_baseline.py --dataset UQ_BPIC13I
  python hawkes_baseline.py --all --backend auto
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# np.trapz was renamed np.trapezoid in numpy 2.0
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

HERE = Path(__file__).resolve().parent                 # evaluation/
REPO_ROOT = HERE.parent                                # THP_for_PPM/
DATA_DIR = REPO_ROOT / "data"
JEA_DIR = REPO_ROOT / "paper_results" / "per_dataset"
RES_UNI = REPO_ROOT / "results_hawkes_v2"
RES_MK = REPO_ROOT / "results_hawkes_marked_v2"

SECONDS_PER_DAY = 86_400.0
FOLD = 0

# Same grid as the THP evaluation, so the tolerances are comparable.
EPS_GRID = np.geomspace(1e-6, 5.0, 25)  # days

UQ_DATASETS = [
    "UQ_HelpDesk", "UQ_Sepsis", "UQ_BPIC13I", "UQ_BPIC20DD",
    "UQ_BPIC20RFP", "UQ_BPIC20PTC", "UQ_BPIC20ID", "UQ_BPIC20TPD",
    "UQ_BPIC12", "UQ_BPIC15_1",
]

# Decay grid (per-day). Logs span sub-minute (BPIC12) to multi-day gaps, so the
# excitation timescale 1/β must be searched wide.
BETA_GRID = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4])

# Cap on the oracle-length RT rollout (bounds cost on very long traces).
MAX_ROLLOUT = 200


# ===========================================================================
# Data loading  (mirror preprocess/dataset.py, CSV-only, no torch)
# ===========================================================================
def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    return df


def load_cases(path: Path, vocab: dict[str, int], unk: int):
    """Return a list of per-case dicts from a split CSV.

    Each case: {t: float[n] days since case start (sorted),
                m: int[n] activity index (vocab, unseen->unk),
                rt: float[n] remaining time (days) to CaseEndTime}.
    Cases keep their CSV row order within a CaseID (already time-sorted by the
    THP preprocessing), then are stably sorted by timestamp to be safe.
    """
    df = _read_csv(path)
    cases = []
    for _, g in df.groupby("CaseID", sort=False):
        g = g.sort_values("Timestamp", kind="stable")
        ts = g["Timestamp"].values.astype("datetime64[ns]")
        t0 = ts[0]
        t_days = (ts - t0) / np.timedelta64(1, "s") / SECONDS_PER_DAY
        t_days = t_days.astype(float)
        m = np.array([vocab.get(a, unk) for a in g["Activity"]], dtype=int)
        if "CaseEndTime" in g.columns:
            ce = pd.to_datetime(g["CaseEndTime"]).values.astype("datetime64[ns]")
            rt = (ce - ts) / np.timedelta64(1, "s") / SECONDS_PER_DAY
        else:
            rt = (ts[-1] - ts) / np.timedelta64(1, "s") / SECONDS_PER_DAY
        cases.append({"t": t_days, "m": m, "rt": rt.astype(float)})
    return cases


def build_vocab(full_train_csv: Path):
    """Activity -> 0-based index from the fit split; UNK = len(vocab)."""
    df = _read_csv(full_train_csv)
    acts = np.unique(df["Activity"].values)
    vocab = {a: i for i, a in enumerate(acts)}
    return vocab, len(acts)  # unk index == len(acts)


# ---------------------------------------------------------------------------
# Tie-breaking jitter:  a classical Hawkes is a *simple* point process (no
# ties; t_{n+1} != t_n). These logs have many tied timestamps (Δt=0, e.g.
# Sepsis ~39%), which are ill-defined for the intensity. We dequantize by
# spreading each group of equal timestamps with a tiny random sub-grid offset.
# Scale = min(0.9·grid, 1 second) where grid = smallest positive Δt; this is
# strictly < grid, so event order is never changed and targets shift by <1s.
# ---------------------------------------------------------------------------
def compute_grid(cases) -> float:
    """Smallest strictly-positive inter-event Δt (days) across the cases."""
    mins = []
    for c in cases:
        d = np.diff(c["t"])
        pos = d[d > 0]
        if pos.size:
            mins.append(float(pos.min()))
    return min(mins) if mins else 1.0 / SECONDS_PER_DAY


def break_ties(t: np.ndarray, scale: float, rng) -> np.ndarray:
    """Return a strictly-increasing copy of sorted `t` by jittering tied groups
    with random offsets in (0, scale). `scale` must be < the gap to the next
    distinct timestamp (guaranteed by scale < grid)."""
    if scale <= 0 or len(t) < 2:
        return t
    out = t.copy()
    n = len(t)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        if j > i:  # a group of (j-i+1) tied timestamps
            offs = np.sort(rng.uniform(0.0, scale, size=j - i + 1))
            out[i:j + 1] = t[i] + offs
        i = j + 1
    return out


# ===========================================================================
# Expected next-event gap from an exponential-kernel intensity
#   λ(t0+s) = μ + β·e^{-β s}·C ,   Λ(s) = μ s + C(1 - e^{-β s})
#   E[S] = ∫₀^∞ e^{-Λ(s)} ds = e^{-C} Σ_{n≥0} C^n / (n! (μ + nβ))
# C = current excitation mass (= α·E for univariate, = 1ᵀA·S for marked).
# ===========================================================================
def e_gap(mu: float, beta: float, C: float, n_terms: int = 400) -> float:
    """Expected waiting time to the next event (days). Closed-form series."""
    mu = max(mu, 1e-12)
    if C <= 1e-12:
        return 1.0 / mu
    if C > 60:  # series unstable for huge excitation -> numeric fallback
        return _e_gap_numeric(mu, beta, C)
    # log-space term accumulation for stability
    total = 0.0
    log_c = math.log(C)
    log_factorial = 0.0
    for n in range(n_terms):
        if n > 0:
            log_factorial += math.log(n)
        log_term = n * log_c - log_factorial - math.log(mu + n * beta)
        term = math.exp(log_term)
        total += term
        if n > 5 and term < 1e-14 * total:
            break
    val = math.exp(-C) * total
    if not math.isfinite(val) or val <= 0:
        return _e_gap_numeric(mu, beta, C)
    return val


def _e_gap_numeric(mu: float, beta: float, C: float) -> float:
    """Numeric ∫₀^∞ e^{-Λ(s)} ds via trapezoid on an adaptive grid."""
    mu = max(mu, 1e-12)
    horizon = max(20.0 / mu, 10.0 / max(beta, 1e-6))
    s = np.linspace(0.0, horizon, 4000)
    Lam = mu * s + C * (1.0 - np.exp(-beta * s))
    surv = np.exp(-Lam)
    return float(_trapz(surv, s))


# ===========================================================================
# Univariate exp-kernel Hawkes  (MLE backend)
# ===========================================================================
def _uni_excitation(t: np.ndarray, beta: float):
    """Per-event excitation E_j = Σ_{i≤j} e^{-β(t_j - t_i)} (recursive)."""
    n = len(t)
    E = np.empty(n)
    e_prev = 0.0
    for j in range(n):
        if j == 0:
            E[j] = 1.0
        else:
            E[j] = 1.0 + math.exp(-beta * (t[j] - t[j - 1])) * e_prev
        e_prev = E[j]
    return E  # E[j] includes the event j itself


def _uni_nll(params, cases, beta):
    """Negative log-likelihood for univariate Hawkes with fixed beta.
    params = (log_mu, log_alpha)."""
    mu = math.exp(params[0])
    alpha = math.exp(params[1])
    nll = 0.0
    for c in cases:
        t = c["t"]
        if len(t) == 0:
            continue
        T = t[-1]
        E = c["_E"]  # precomputed for this beta
        # E_excl_self for intensity at t_j: λ(t_j)=μ + αβ (E_j - 1)
        lam = mu + alpha * beta * (E - 1.0)
        lam = np.maximum(lam, 1e-12)
        nll -= np.sum(np.log(lam))
        # compensator ∫_0^T λ = μ T + α Σ_i (1 - e^{-β(T - t_i)})
        comp = mu * T + alpha * np.sum(1.0 - np.exp(-beta * (T - t)))
        nll += comp
    return nll


def fit_univariate_mle(cases):
    """Fit (μ, α, β) by MLE over a β-grid; return dict of params + LL."""
    best = None
    for beta in BETA_GRID:
        for c in cases:
            c["_E"] = _uni_excitation(c["t"], beta)
        x0 = np.log([1.0, 0.5])
        try:
            res = minimize(_uni_nll, x0, args=(cases, beta), method="L-BFGS-B",
                           options={"maxiter": 200})
            nll = res.fun
            mu, alpha = math.exp(res.x[0]), math.exp(res.x[1])
        except Exception:
            continue
        if best is None or nll < best["nll"]:
            best = {"mu": mu, "alpha": alpha, "beta": float(beta), "nll": float(nll)}
    for c in cases:
        c.pop("_E", None)
    if best is None:
        best = {"mu": 1.0, "alpha": 0.0, "beta": 1.0, "nll": float("inf")}
    return best


# ===========================================================================
# Marked multivariate exp-kernel Hawkes  (MLE backend)
#  Fixed β -> the NLL decomposes per target dimension k (each convex in
#  (μ_k, A_{k,:})≥0). We precompute the excitation matrix Φ (one D-vector per
#  training event, the state *just before* that event) once per β.
# ===========================================================================
def _marked_excitation_matrix(cases, beta, D):
    """Return (Phi, types, Tsum, G):
      Phi[e]  = S-vector (length D) just before event e   (excitation state)
      types[e]= activity index of event e
      Tsum    = Σ_c T_c
      G[d]    = Σ_c Σ_{i: type d} (1 - e^{-β(T_c - t_i)})   (compensator const)
    """
    rows = []
    types = []
    Tsum = 0.0
    G = np.zeros(D)
    for c in cases:
        t, m = c["t"], c["m"]
        n = len(t)
        if n == 0:
            continue
        T = t[-1]
        S = np.zeros(D)
        prev = t[0]
        for i in range(n):
            if i > 0:
                S *= math.exp(-beta * (t[i] - prev))
                prev = t[i]
            rows.append(S.copy())      # state just before event i
            types.append(m[i])
            S[m[i]] += 1.0             # event i joins the excitation
        Tsum += T
        # compensator constant per dim: scatter (1 - e^{-β(T-t_i)}) into G[type_i]
        np.add.at(G, m, 1.0 - np.exp(-beta * (T - t)))
    Phi = np.asarray(rows) if rows else np.zeros((0, D))
    return Phi, np.asarray(types, dtype=int), Tsum, G


def _fit_dim_k(phi_k, G, Tsum, beta, D):
    """Convex per-dim fit: min over (μ_k≥0, a=A_{k,:}≥0)
        -Σ_rows log(μ_k + β·a·φ) + μ_k·Tsum + a·G .
    phi_k = excitation rows at type-k events (n_k × D). Returns (μ_k, a[D])."""
    nk = phi_k.shape[0]
    if nk == 0:
        return 1e-6, np.zeros(D)
    # Reduce dimensionality: only columns active at type-k events matter for the
    # log term; others are pinned by the linear penalty G -> 0.
    active = np.where(phi_k.sum(0) > 0)[0]
    Gd = G[active]
    phi = phi_k[:, active]

    def nll(x):
        mu = x[0]
        a = x[1:]
        lam = mu + beta * (phi @ a)
        lam = np.maximum(lam, 1e-12)
        return -np.sum(np.log(lam)) + mu * Tsum + a @ Gd

    def grad(x):
        mu = x[0]
        a = x[1:]
        lam = np.maximum(mu + beta * (phi @ a), 1e-12)
        inv = 1.0 / lam
        gmu = -np.sum(inv) + Tsum
        ga = -beta * (phi.T @ inv) + Gd
        return np.concatenate([[gmu], ga])

    x0 = np.concatenate([[max(nk / max(Tsum, 1e-6), 1e-6)], np.full(len(active), 0.01)])
    bounds = [(1e-9, None)] * (len(active) + 1)
    try:
        res = minimize(nll, x0, jac=grad, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300})
        mu_k = float(res.x[0])
        a_full = np.zeros(D)
        a_full[active] = res.x[1:]
    except Exception:
        mu_k = max(nk / max(Tsum, 1e-6), 1e-6)
        a_full = np.zeros(D)
    return mu_k, a_full


def fit_marked_mle(cases, D, max_dim_full=120, verbose=False):
    """Fit μ (D,), A (D,D), shared β by MLE over a β-grid.

    For D>max_dim_full the full D×D adjacency is impractical/unstable (e.g.
    BPIC15_1, D≈382): fall back to a structured fit — per-type baseline μ_k +
    a single shared scalar self/cross excitation a (A = a·11ᵀ). Flagged in
    the returned dict via `structured=True`.
    """
    structured = D > max_dim_full
    best = None
    for beta in BETA_GRID:
        Phi, types, Tsum, G = _marked_excitation_matrix(cases, beta, D)
        if Phi.shape[0] == 0:
            continue
        if structured:
            mu, A, nll = _fit_structured(Phi, types, Tsum, G, beta, D)
        else:
            mu = np.zeros(D)
            A = np.zeros((D, D))
            for k in range(D):
                rows_k = Phi[types == k]
                mu_k, a_k = _fit_dim_k(rows_k, G, Tsum, beta, D)
                mu[k] = mu_k
                A[k] = a_k
            # total NLL for model selection
            lam = mu[types] + beta * np.einsum("ed,ed->e", A[types], Phi)
            lam = np.maximum(lam, 1e-12)
            nll = float(-np.sum(np.log(lam)) + mu.sum() * Tsum + np.sum(A * G[None, :]))
        if best is None or nll < best["nll"]:
            best = {"mu": mu, "A": A, "beta": float(beta), "nll": float(nll),
                    "structured": structured}
        if verbose:
            print(f"      [marked] beta={beta:<8g} nll={nll:.1f}")
    if best is None:
        best = {"mu": np.full(D, 1e-6), "A": np.zeros((D, D)), "beta": 1.0,
                "nll": float("inf"), "structured": structured}
    return best


def _fit_structured(Phi, types, Tsum, G, beta, D):
    """A = a·11ᵀ (single scalar excitation) + per-type μ_k. Convex in (μ, a)."""
    tot = Phi.sum(1)                       # 1ᵀ S  per event
    Gsum = G.sum()
    counts = np.bincount(types, minlength=D)

    def nll(x):
        mu = x[:D]
        a = x[D]
        lam = mu[types] + beta * a * tot
        lam = np.maximum(lam, 1e-12)
        return -np.sum(np.log(lam)) + mu.sum() * Tsum + a * Gsum

    def grad(x):
        mu = x[:D]
        a = x[D]
        lam = np.maximum(mu[types] + beta * a * tot, 1e-12)
        inv = 1.0 / lam
        gmu = -np.bincount(types, weights=inv, minlength=D) + Tsum
        ga = -beta * np.sum(tot * inv) + Gsum
        return np.concatenate([gmu, [ga]])

    x0 = np.concatenate([np.maximum(counts / max(Tsum, 1e-6), 1e-6), [0.001]])
    bounds = [(1e-9, None)] * (D + 1)
    res = minimize(nll, x0, jac=grad, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300})
    mu = res.x[:D]
    A = np.full((D, D), res.x[D])
    return mu, A, float(res.fun)


# ===========================================================================
# tick backend (optional)
# ===========================================================================
def _try_tick():
    try:
        import tick.hawkes  # noqa: F401
        return True
    except Exception:
        return False


def fit_univariate_tick(cases):
    from tick.hawkes import HawkesExpKern
    best = None
    realiz = [[np.ascontiguousarray(np.sort(c["t"]))] for c in cases if len(c["t"]) > 1]
    for beta in BETA_GRID:
        try:
            learner = HawkesExpKern(decays=beta, penalty="l2", C=1e3)
            learner.fit(realiz)
            mu = float(np.asarray(learner.baseline).ravel()[0])
            alpha = float(np.asarray(learner.adjacency).ravel()[0])
            score = learner.score()
        except Exception:
            continue
        if best is None or score > best["nll_neg"]:
            best = {"mu": max(mu, 1e-9), "alpha": max(alpha, 0.0),
                    "beta": float(beta), "nll_neg": score}
    if best is None:
        raise RuntimeError("tick univariate fit failed")
    best["nll"] = -best.pop("nll_neg")
    return best


def fit_marked_tick(cases, D, beta_grid=BETA_GRID):
    from tick.hawkes import HawkesExpKern
    best = None
    realiz = []
    for c in cases:
        if len(c["t"]) < 2:
            continue
        dims = [np.ascontiguousarray(np.sort(c["t"][c["m"] == d])) for d in range(D)]
        realiz.append(dims)
    for beta in beta_grid:
        try:
            learner = HawkesExpKern(decays=beta, penalty="l1", C=1e3)
            learner.fit(realiz)
            mu = np.asarray(learner.baseline, dtype=float).ravel()
            A = np.asarray(learner.adjacency, dtype=float)
            score = learner.score()
        except Exception:
            continue
        if best is None or score > best["nll_neg"]:
            best = {"mu": np.maximum(mu, 1e-9), "A": np.maximum(A, 0.0),
                    "beta": float(beta), "nll_neg": score, "structured": False}
    if best is None:
        raise RuntimeError("tick marked fit failed")
    best["nll"] = -best.pop("nll_neg")
    return best


# ===========================================================================
# Prediction
# ===========================================================================
def predict_univariate(model, cases):
    """Return (derr, pls) for next-event time and (rt_err) per event."""
    mu, alpha, beta = model["mu"], model["alpha"], model["beta"]
    derr, pls, rt_err = [], [], []
    for c in cases:
        t, rt = c["t"], c["rt"]
        n = len(t)
        if n == 0:
            continue
        E = _uni_excitation(t, beta)       # E[j] excitation incl. event j
        # next-event time prediction at each prefix end j (predict event j+1)
        for j in range(n - 1):
            C = alpha * E[j]
            pred_dt = e_gap(mu, beta, C)
            gdt = t[j + 1] - t[j]
            derr.append(abs(pred_dt - gdt))
            pls.append(j + 1)
        # oracle-length remaining-time rollout for every event j
        for j in range(n):
            L = min(n - 1 - j, MAX_ROLLOUT)
            Ej = E[j]
            tot = 0.0
            for _ in range(L):
                C = alpha * Ej
                g = e_gap(mu, beta, C)
                tot += g
                Ej = 1.0 + math.exp(-beta * g) * Ej   # add predicted event
            rt_err.append(abs(tot - rt[j]))
    return np.asarray(derr), np.asarray(pls, dtype=int), np.asarray(rt_err)


def predict_marked(model, cases, D):
    """Return arrays for JEA (cor, derr, ranks, pls) + rt_err + activity acc."""
    mu, A, beta = model["mu"], model["A"], model["beta"]
    cor, derr, ranks, pls, rt_err = [], [], [], [], []
    AS_one = A.sum(0)                      # 1ᵀA  (length D): C = AS_one · S
    for c in cases:
        t, m = c["t"], c["m"]
        rt = c["rt"]
        n = len(t)
        if n == 0:
            continue
        # roll the excitation state S forward over the true sequence
        S = np.zeros(D)
        states = []                        # state just before each event
        prev = t[0]
        for i in range(n):
            if i > 0:
                S = S * math.exp(-beta * (t[i] - prev))
                prev = t[i]
            states.append(S.copy())
            S[m[i]] += 1.0
        for j in range(n - 1):
            Sj = states[j].copy()
            Sj[m[j]] += 1.0                # include event j (prefix end)
            C = float(AS_one @ Sj)
            pred_dt = e_gap(mu.sum(), beta, C)
            # next mark = which dim fires first = argmax of the imminent intensity
            # λ_k(t0⁺) = μ_k + β·(A·S) (full excitation; the shared e^{-βs} factor
            # cancels in the argmax, so evaluate at s→0⁺ rather than at E[Δt]).
            lam_k = mu + beta * (A @ Sj)
            pred_mark = int(np.argmax(lam_k))
            gt = int(m[j + 1])
            cor.append(int(pred_mark == gt))
            derr.append(abs(pred_dt - (t[j + 1] - t[j])))
            ranks.append(int(np.sum(lam_k > lam_k[gt]) + 1))
            pls.append(j + 1)
        # oracle-length RT rollout
        for j in range(n):
            L = min(n - 1 - j, MAX_ROLLOUT)
            Sj = states[j].copy()
            Sj[m[j]] += 1.0
            tot = 0.0
            for _ in range(L):
                C = float(AS_one @ Sj)
                g = e_gap(mu.sum(), beta, C)
                tot += g
                lam_k = mu + beta * (A @ Sj)   # imminent intensity (s→0⁺)
                pm = int(np.argmax(lam_k))
                Sj = Sj * math.exp(-beta * g)
                Sj[pm] += 1.0
            rt_err.append(abs(tot - rt[j]))
    acc = float(np.mean(cor)) if cor else 0.0
    return (np.asarray(cor), np.asarray(derr), np.asarray(ranks, dtype=int),
            np.asarray(pls, dtype=int), np.asarray(rt_err), acc)


# ===========================================================================
# JEA finalize (same schema as run_evaluation_uq._finalize_from_arrays)
# ===========================================================================
def finalize(correct, dt_err, ranks, prefix_lens, num_types):
    correct = np.asarray(correct, dtype=bool)
    dt_err = np.asarray(dt_err, dtype=float)
    per_event = [float((correct & (dt_err < eps)).mean()) for eps in EPS_GRID]
    eps_mid = EPS_GRID[len(EPS_GRID) // 2]
    pl = np.asarray(prefix_lens, dtype=int)
    per_prefix = {}
    for k in sorted(set(pl.tolist())):
        msk = pl == k
        per_prefix[str(int(k))] = float((correct[msk] & (dt_err[msk] < eps_mid)).mean())
    out = {
        "per_event_teacher": per_event,
        "activity_accuracy_teacher": float(correct.mean()) * 100,
        "time_mae_teacher": float(dt_err.mean()),
        "per_prefix_teacher": per_prefix,
    }
    if ranks is not None and num_types:
        ranks = np.asarray(ranks, dtype=int)
        topk = {}
        for ei, eps in enumerate(EPS_GRID):
            time_ok = dt_err < eps
            topk[str(ei)] = [float(((ranks <= k) & time_ok).mean() * 100)
                             for k in range(1, num_types + 1)]
        out["topk_jea"] = topk
        out["num_types"] = int(num_types)
    return out


# ===========================================================================
# Per-dataset driver
# ===========================================================================
def write_txt(path: Path, ll, mae, mae_rt, acc):
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%d.%m.%y-%H.%M")
    path.write_text(f"Starting time: {stamp}\n"
                    f"TEST  ll={ll:.5f}  mae={mae:.5f}  mae_rt={mae_rt:.5f}  acc={acc:.5f}\n")


def run_dataset(ds, backend="mle", jitter=True, verbose=True):
    fold_file = f"fold{FOLD}_variation0_{ds}"
    full_train = DATA_DIR / f"full_train_{fold_file}.csv"
    test_csv = DATA_DIR / f"test_{fold_file}.csv"
    if not full_train.exists() or not test_csv.exists():
        print(f"  [{ds}] SKIP — missing {full_train.name} or {test_csv.name}")
        return

    vocab, D = build_vocab(full_train)
    unk = D                                 # unseen test act -> UNK dim
    D_eff = D + 1                            # include UNK dim in the process
    train_cases = load_cases(full_train, vocab, unk)
    test_cases = load_cases(test_csv, vocab, unk)
    n_train_ev = sum(len(c["t"]) for c in train_cases)

    # break ties so the point process is simple (t_{n+1} != t_n)
    tie_scale = 0.0
    if jitter:
        grid = compute_grid(train_cases)
        tie_scale = min(0.9 * grid, 1.0 / SECONDS_PER_DAY)   # < grid, ≤ 1 second
        rng = np.random.default_rng(0)
        for c in train_cases + test_cases:
            c["t"] = break_ties(c["t"], tie_scale, rng)
    print(f"  [{ds}] D={D} (+UNK), train cases={len(train_cases)} "
          f"events={n_train_ev}, test cases={len(test_cases)}, "
          f"tie_jitter={tie_scale * SECONDS_PER_DAY:.4g}s")

    use_tick = backend == "tick" or (backend == "auto" and _try_tick())
    eng = "tick" if use_tick else "mle"

    # ---- fit ----
    t_fit0 = time.time()
    # univariate
    if use_tick:
        try:
            uni = fit_univariate_tick(train_cases)
        except Exception as e:
            print(f"    tick uni failed ({e}); falling back to MLE")
            uni = fit_univariate_mle(train_cases); eng = "mle(uni-fallback)"
    else:
        uni = fit_univariate_mle(train_cases)
    # marked
    if use_tick:
        try:
            mk = fit_marked_tick(train_cases, D_eff)
        except Exception as e:
            print(f"    tick marked failed ({e}); falling back to MLE")
            mk = fit_marked_mle(train_cases, D_eff, verbose=verbose)
    else:
        mk = fit_marked_mle(train_cases, D_eff, verbose=verbose)
    fit_min = (time.time() - t_fit0) / 60.0
    print(f"    fit[{eng}] uni(mu={uni['mu']:.3g},a={uni['alpha']:.3g},b={uni['beta']:g}) "
          f"marked(beta={mk['beta']:g}{',structured' if mk.get('structured') else ''}) "
          f"in {fit_min:.2f} min")

    # ---- predict ----
    u_derr, u_pls, u_rt = predict_univariate(uni, test_cases)
    m_cor, m_derr, m_rank, m_pls, m_rt, m_acc = predict_marked(mk, test_cases, D_eff)

    uni_mae = float(u_derr.mean()) if u_derr.size else float("nan")
    uni_mae_rt = float(u_rt.mean()) if u_rt.size else float("nan")
    mk_mae = float(m_derr.mean()) if m_derr.size else float("nan")
    mk_mae_rt = float(m_rt.mean()) if m_rt.size else float("nan")
    print(f"    uni:    mae={uni_mae:.4f}  mae_rt={uni_mae_rt:.4f}")
    print(f"    marked: mae={mk_mae:.4f}  mae_rt={mk_mae_rt:.4f}  acc={m_acc*100:.1f}%")

    # ---- write txt (compare_all) ----
    write_txt(RES_UNI / f"{fold_file}_hawkes.txt", -uni["nll"], uni_mae, uni_mae_rt, 0.0)
    write_txt(RES_MK / f"{fold_file}_hawkes_marked.txt", -mk["nll"], mk_mae, mk_mae_rt, m_acc)

    # ---- write JEA json ----
    uni_method = {
        "per_event_teacher": [0.0 for _ in EPS_GRID],
        "activity_accuracy_teacher": 0.0,
        "time_mae_teacher": uni_mae,
        "per_prefix_teacher": {str(int(k)): 0.0 for k in sorted(set(u_pls.tolist()))},
        "no_activity": True,
    }
    mk_method = finalize(m_cor, m_derr, m_rank, m_pls, D_eff)
    out = {
        "config": {"fold": FOLD, "dataset": ds, "eps_grid": EPS_GRID.tolist(),
                   "backend": eng, "structured_marked": bool(mk.get("structured", False)),
                   "tie_jitter_sec": tie_scale * SECONDS_PER_DAY},
        "methods": {"Hawkes (uni)": uni_method, "Hawkes (marked)": mk_method},
    }
    jdir = JEA_DIR / ds
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "hawkes_results.json").write_text(json.dumps(out, indent=2))

    # ---- timing log (collect_timings reads 'Train ... t=<min>min') ----
    (HERE / f"logs_hawkes_v2_{ds}.log").write_text(
        f"Hawkes fit [{eng}] dataset={ds}\n"
        f"Train epoch 1 t={fit_min:.4f}min (MLE fit, single pass)\n")
    print(f"    saved txt + hawkes_results.json + log")


def main():
    ap = argparse.ArgumentParser(description="Traditional Hawkes baselines for UQ4PPM")
    ap.add_argument("--dataset", "-d", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--backend", choices=["mle", "tick", "auto"], default="mle")
    ap.add_argument("--no-jitter", dest="jitter", action="store_false",
                    help="disable tie-breaking jitter (default: on, so t_{n+1} != t_n)")
    args = ap.parse_args()

    if args.all:
        targets = UQ_DATASETS
    elif args.dataset:
        targets = [args.dataset]
    else:
        ap.error("pass --dataset <DS> or --all")

    print(f"Hawkes baseline — backend={args.backend} — {len(targets)} dataset(s)")
    for ds in targets:
        print(f"\n=== {ds} ===")
        try:
            run_dataset(ds, backend=args.backend, jitter=args.jitter)
        except Exception as e:
            import traceback
            print(f"  [{ds}] ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

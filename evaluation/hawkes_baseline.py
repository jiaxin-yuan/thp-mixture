#!/usr/bin/env python3
"""Parametric exponential-kernel Hawkes baselines (numpy/scipy MLE) for the
UQ4PPM comparison.

  Hawkes (uni)     λ(t) = μ + Σ_{t_i<t} α·β·e^{-β(t-t_i)}; times only, cf. THP-B
  Hawkes (marked)  λ_k(t) = μ_k + Σ_j Σ_{t_i^j<t} A_{kj}·β·e^{-β(t-t_i^j)}, one
                   dim per activity, predicts activity and time, cf. THP-M

Reads the strict-split CSVs and follows the target conventions of
preprocess/dataset.py (all times in days): Δt = consecutive timestamp diff
within a case, remaining_time = (CaseEndTime - Timestamp)/86400, vocabulary
from full_train (unseen test activity -> UNK). Scored per position over the
prefixes [2, n-1] of each test case. Next event time is E[Δt] = ∫₀^∞ e^{-Λ(s)}ds
(e_gap), next activity argmax_k λ_k, remaining time an oracle-length greedy
rollout, a Hawkes having no notion of case end.

Writes results_hawkes_v2/*.txt, results_hawkes_marked_v2/*.txt,
paper_results/per_dataset/<DS>/hawkes_results.json and logs_hawkes_v2_<DS>.log.

Usage (numpy / pandas / scipy only, so the thppm env of environment.yml):
  python hawkes_baseline.py --dataset UQ_BPIC13I
  python hawkes_baseline.py --all
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
RESULTS_DIR = REPO_ROOT / "paper_results" / "per_dataset"
RES_UNI = REPO_ROOT / "results_hawkes_v2"
RES_MK = REPO_ROOT / "results_hawkes_marked_v2"

SECONDS_PER_DAY = 86_400.0
FOLD = 0

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


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    return df


def load_cases(path: Path, vocab: dict[str, int], unk: int):
    """Per-case dicts {t: days since case start, m: activity index, rt: remaining days} from a split CSV."""
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
    return vocab, len(acts)


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
    """Strictly-increasing copy of sorted `t`, spreading each tied group over
    (0, scale). A classical Hawkes is a simple point process, but these logs
    have many tied timestamps (Sepsis ~39%) at which the intensity is
    ill-defined. scale < grid keeps the event order and shifts targets by <1s.
    """
    if scale <= 0 or len(t) < 2:
        return t
    out = t.copy()
    n = len(t)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and t[j + 1] == t[i]:
            j += 1
        if j > i:
            offs = np.sort(rng.uniform(0.0, scale, size=j - i + 1))
            out[i:j + 1] = t[i] + offs
        i = j + 1
    return out


def e_gap(mu: float, beta: float, C: float, n_terms: int = 400) -> float:
    """E[S] = ∫₀^∞ e^{-Λ(s)}ds in days, for λ(t0+s) = μ + β·e^{-βs}·C, by the
    series e^{-C} Σ_n C^n / (n!(μ+nβ)); C is the excitation mass (α·E univariate,
    1ᵀA·S marked)."""
    mu = max(mu, 1e-12)
    if C <= 1e-12:
        return 1.0 / mu
    if C > 60:  # series unstable for huge excitation -> numeric fallback
        return _e_gap_numeric(mu, beta, C)
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


def _uni_excitation(t: np.ndarray, beta: float):
    """Per-event excitation E_j = Σ_{i≤j} e^{-β(t_j - t_i)}, event j included."""
    n = len(t)
    E = np.empty(n)
    e_prev = 0.0
    for j in range(n):
        if j == 0:
            E[j] = 1.0
        else:
            E[j] = 1.0 + math.exp(-beta * (t[j] - t[j - 1])) * e_prev
        e_prev = E[j]
    return E


def _uni_nll(params, cases, beta):
    """NLL of the univariate Hawkes at fixed beta; params = (log_mu, log_alpha)."""
    mu = math.exp(params[0])
    alpha = math.exp(params[1])
    nll = 0.0
    for c in cases:
        t = c["t"]
        if len(t) == 0:
            continue
        T = t[-1]
        E = c["_E"]  # precomputed for this beta
        lam = mu + alpha * beta * (E - 1.0)
        lam = np.maximum(lam, 1e-12)
        nll -= np.sum(np.log(lam))
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


def _marked_excitation_matrix(cases, beta, D):
    """Excitation state just before each training event: (Phi [E, D], types [E], Tsum = Σ_c T_c, G[d] = Σ_c Σ_{i: type d} (1 - e^{-β(T_c-t_i)}))."""
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
            rows.append(S.copy())
            types.append(m[i])
            S[m[i]] += 1.0
        Tsum += T
        np.add.at(G, m, 1.0 - np.exp(-beta * (T - t)))
    Phi = np.asarray(rows) if rows else np.zeros((0, D))
    return Phi, np.asarray(types, dtype=int), Tsum, G


def _fit_dim_k(phi_k, G, Tsum, beta, D):
    """Convex per-dim fit of (μ_k≥0, a=A_{k,:}≥0) minimizing -Σ_rows log(μ_k + β·a·φ) + μ_k·Tsum + a·G, over the type-k excitation rows phi_k [n_k, D]."""
    nk = phi_k.shape[0]
    if nk == 0:
        return 1e-6, np.zeros(D)
    # only columns active at type-k events enter the log term; the rest are
    # pinned to 0 by the linear penalty G
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
    """Fit μ (D,), A (D,D), shared β by MLE over a β-grid, the NLL decomposing
    per target dimension at fixed β. Above max_dim_full the full adjacency is
    unstable (BPIC15_1, D≈382), so a structured fit (μ_k + scalar A = a·11ᵀ)
    takes over, flagged as `structured` in the returned dict.
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


def predict_univariate(model, cases):
    """Return the per-event next event time errors and remaining time errors."""
    mu, alpha, beta = model["mu"], model["alpha"], model["beta"]
    derr, rt_err = [], []
    for c in cases:
        t, rt = c["t"], c["rt"]
        n = len(t)
        if n == 0:
            continue
        E = _uni_excitation(t, beta)
        for j in range(n - 1):
            C = alpha * E[j]
            pred_dt = e_gap(mu, beta, C)
            gdt = t[j + 1] - t[j]
            derr.append(abs(pred_dt - gdt))
        for j in range(n):
            L = min(n - 1 - j, MAX_ROLLOUT)
            Ej = E[j]
            tot = 0.0
            for _ in range(L):
                C = alpha * Ej
                g = e_gap(mu, beta, C)
                tot += g
                Ej = 1.0 + math.exp(-beta * g) * Ej   # add the predicted event
            rt_err.append(abs(tot - rt[j]))
    return np.asarray(derr), np.asarray(rt_err)


def predict_marked(model, cases, D):
    """Return the per-event activity hits, time errors, remaining time errors and the activity accuracy."""
    mu, A, beta = model["mu"], model["A"], model["beta"]
    cor, derr, rt_err = [], [], []
    AS_one = A.sum(0)                      # 1ᵀA  (length D): C = AS_one · S
    for c in cases:
        t, m = c["t"], c["m"]
        rt = c["rt"]
        n = len(t)
        if n == 0:
            continue
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
            # next mark = the dim that fires first; the shared e^{-βs} cancels in
            # the argmax, so λ_k is evaluated at s->0⁺ rather than at E[Δt]
            lam_k = mu + beta * (A @ Sj)
            pred_mark = int(np.argmax(lam_k))
            gt = int(m[j + 1])
            cor.append(int(pred_mark == gt))
            derr.append(abs(pred_dt - (t[j + 1] - t[j])))
        for j in range(n):
            L = min(n - 1 - j, MAX_ROLLOUT)
            Sj = states[j].copy()
            Sj[m[j]] += 1.0
            tot = 0.0
            for _ in range(L):
                C = float(AS_one @ Sj)
                g = e_gap(mu.sum(), beta, C)
                tot += g
                lam_k = mu + beta * (A @ Sj)
                pm = int(np.argmax(lam_k))
                Sj = Sj * math.exp(-beta * g)
                Sj[pm] += 1.0
            rt_err.append(abs(tot - rt[j]))
    acc = float(np.mean(cor)) if cor else 0.0
    return np.asarray(cor), np.asarray(derr), np.asarray(rt_err), acc


def finalize(correct, dt_err):
    """Same schema as run_evaluation_uq._finalize_from_arrays."""
    correct = np.asarray(correct, dtype=bool)
    dt_err = np.asarray(dt_err, dtype=float)
    return {
        "activity_accuracy_teacher": float(correct.mean()) * 100,
        "time_mae_teacher": float(dt_err.mean()),
    }


def write_txt(path: Path, ll, mae, mae_rt, acc):
    """Write the one-line result file the compare_all aggregator reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%d.%m.%y-%H.%M")
    path.write_text(f"Starting time: {stamp}\n"
                    f"TEST  ll={ll:.5f}  mae={mae:.5f}  mae_rt={mae_rt:.5f}  acc={acc:.5f}\n")


def run_dataset(ds, jitter=True, verbose=True):
    """Fit both Hawkes models on one dataset, score the test split, write the outputs."""
    fold_file = f"fold{FOLD}_variation0_{ds}"
    full_train = DATA_DIR / f"full_train_{fold_file}.csv"
    test_csv = DATA_DIR / f"test_{fold_file}.csv"
    if not full_train.exists() or not test_csv.exists():
        print(f"  [{ds}] SKIP — missing {full_train.name} or {test_csv.name}")
        return

    vocab, D = build_vocab(full_train)
    unk = D                                  # unseen test act -> UNK dim
    D_eff = D + 1
    train_cases = load_cases(full_train, vocab, unk)
    test_cases = load_cases(test_csv, vocab, unk)
    n_train_ev = sum(len(c["t"]) for c in train_cases)

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

    eng = "mle"

    t_fit0 = time.time()
    uni = fit_univariate_mle(train_cases)
    mk = fit_marked_mle(train_cases, D_eff, verbose=verbose)
    fit_min = (time.time() - t_fit0) / 60.0
    print(f"    fit[{eng}] uni(mu={uni['mu']:.3g},a={uni['alpha']:.3g},b={uni['beta']:g}) "
          f"marked(beta={mk['beta']:g}{',structured' if mk.get('structured') else ''}) "
          f"in {fit_min:.2f} min")

    u_derr, u_rt = predict_univariate(uni, test_cases)
    m_cor, m_derr, m_rt, m_acc = predict_marked(mk, test_cases, D_eff)

    uni_mae = float(u_derr.mean()) if u_derr.size else float("nan")
    uni_mae_rt = float(u_rt.mean()) if u_rt.size else float("nan")
    mk_mae = float(m_derr.mean()) if m_derr.size else float("nan")
    mk_mae_rt = float(m_rt.mean()) if m_rt.size else float("nan")
    print(f"    uni:    mae={uni_mae:.4f}  mae_rt={uni_mae_rt:.4f}")
    print(f"    marked: mae={mk_mae:.4f}  mae_rt={mk_mae_rt:.4f}  acc={m_acc*100:.1f}%")

    write_txt(RES_UNI / f"{fold_file}_hawkes.txt", -uni["nll"], uni_mae, uni_mae_rt, 0.0)
    write_txt(RES_MK / f"{fold_file}_hawkes_marked.txt", -mk["nll"], mk_mae, mk_mae_rt, m_acc)

    uni_method = {
        "activity_accuracy_teacher": 0.0,
        "time_mae_teacher": uni_mae,
        "no_activity": True,           # H-uni models times only, so Act is undefined
    }
    mk_method = finalize(m_cor, m_derr)
    out = {
        "config": {"fold": FOLD, "dataset": ds,
                   "backend": eng, "structured_marked": bool(mk.get("structured", False)),
                   "tie_jitter_sec": tie_scale * SECONDS_PER_DAY},
        "methods": {"Hawkes (uni)": uni_method, "Hawkes (marked)": mk_method},
    }
    jdir = RESULTS_DIR / ds
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "hawkes_results.json").write_text(json.dumps(out, indent=2))

    # collect_timings reads 'Train ... t=<min>min'
    (HERE / f"logs_hawkes_v2_{ds}.log").write_text(
        f"Hawkes fit [{eng}] dataset={ds}\n"
        f"Train epoch 1 t={fit_min:.4f}min (MLE fit, single pass)\n")
    print(f"    saved txt + hawkes_results.json + log")


def main():
    ap = argparse.ArgumentParser(description="Traditional Hawkes baselines for UQ4PPM")
    ap.add_argument("--dataset", "-d", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-jitter", dest="jitter", action="store_false",
                    help="disable tie-breaking jitter (default: on, so t_{n+1} != t_n)")
    args = ap.parse_args()

    if args.all:
        targets = UQ_DATASETS
    elif args.dataset:
        targets = [args.dataset]
    else:
        ap.error("pass --dataset <DS> or --all")

    print(f"Hawkes baseline (MLE) — {len(targets)} dataset(s)")
    for ds in targets:
        print(f"\n=== {ds} ===")
        try:
            run_dataset(ds, jitter=args.jitter)
        except Exception as e:
            import traceback
            print(f"  [{ds}] ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

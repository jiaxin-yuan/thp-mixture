"""Generative-evaluation utilities for the zero-inflated LogNormal mixture (ZILM).

Parametrization matches THP_for_PPM/Utils.py (mixture_time_loss / expected_dt):
    pi    = sigmoid(tie_logit)                          P(dt = 0)
    w     = softmax(time_params[..., :K])               mixture weights
    mu    = time_params[..., K:2K]                      log-space means
    sigma = softplus(time_params[..., 2K:]) + 1e-4      log-space stds
Law:  dt = 0 w.p. pi;  dt ~ sum_k w_k LogNormal(mu_k, sigma_k) w.p. 1 - pi.

All functions operate on flat per-event tensors:
    pi [N], log_w/mu/sigma [N, K], dt [N]; candidate sets are [N, C].

Provides:
    extract_params      head outputs -> (pi, log_w, mu, sigma)
    time_nll            per-event NLL of the mixed law (nats; density in days^-1)
    cont_cdf            CDF of the continuous (LogNormal-mixture) part
    zilm_cdf            full CDF  F(t) = pi + (1-pi) * cont_cdf(t)
    zilm_mean           E[dt]  (the legacy decode, kept as ablation)
    quantile            vectorized quantile via bisection (median = quantile(0.5))
    eps_ball_mass       P(|dt - t| < eps) including the atom at 0
    eps_mode            decode t maximizing the eps-ball mass (JEA-optimal point)
    pit                 randomized PIT for the mixed discrete-continuous law
    ks_uniform          Kolmogorov-Smirnov distance of PIT values from U(0,1)
    interval_coverage   empirical coverage + mean width of central intervals
    ece                 expected calibration error of a categorical head
    mark_nll            per-event cross-entropy of the mark head
    risk_coverage       selective-prediction curve, AURC, accuracy@coverage
"""

import math

import torch
import torch.nn.functional as F

_SQRT2 = math.sqrt(2.0)
_LOG_2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Parameters and likelihood
# --------------------------------------------------------------------------- #

def extract_params(tie_logit, time_params, K):
    """Head outputs -> (pi [.], log_w [., K], mu [., K], sigma [., K])."""
    pi = torch.sigmoid(tie_logit)
    log_w = F.log_softmax(time_params[..., :K], dim=-1)
    mu = time_params[..., K:2 * K]
    sigma = F.softplus(time_params[..., 2 * K:]) + 1e-4
    return pi, log_w, mu, sigma


def time_nll(dt, pi, log_w, mu, sigma):
    """Per-event NLL [N] of the ZILM (tie: -log pi; else -log(1-pi) - log mix-pdf).

    Mixed discrete-continuous law: tie events contribute a log-probability,
    non-tie events a log-density (days^-1) — same convention as
    Utils.mixture_time_loss, which this matches term-for-term (but per event).
    """
    log_pi = torch.log(pi.clamp(min=1e-12))
    log_1mpi = torch.log((1.0 - pi).clamp(min=1e-12))
    log_dt = torch.log(dt.clamp(min=1e-8)).unsqueeze(-1)            # [N, 1]
    log_pdf = (-log_dt - torch.log(sigma) - 0.5 * _LOG_2PI
               - 0.5 * ((log_dt - mu) / sigma) ** 2)                # [N, K]
    log_mix = torch.logsumexp(log_w + log_pdf, dim=-1)              # [N]
    return torch.where(dt == 0, -log_pi, -log_1mpi - log_mix)


def cont_cdf(t, log_w, mu, sigma):
    """CDF of the continuous part at t (>0); 0 for t <= 0.

    t may be [N] (one point per event) or [N, C] (C candidates per event,
    with params [N, K]).
    """
    if t.dim() == mu.dim():                       # [N, C] vs [N, K]
        log_w, mu, sigma = (x.unsqueeze(-2) for x in (log_w, mu, sigma))
    z = (torch.log(t.clamp(min=1e-12)).unsqueeze(-1) - mu) / sigma
    phi = 0.5 * (1.0 + torch.erf(z / _SQRT2))
    c = (log_w.exp() * phi).sum(-1)
    return torch.where(t > 0, c, torch.zeros_like(c))


def zilm_cdf(t, pi, log_w, mu, sigma):
    """Full CDF F(t) = pi * 1[t >= 0] + (1 - pi) * cont_cdf(t)."""
    c = cont_cdf(t, log_w, mu, sigma)
    if t.dim() > pi.dim():
        pi = pi.unsqueeze(-1)
    return torch.where(t >= 0, pi + (1.0 - pi) * c, torch.zeros_like(c))


def zilm_mean(pi, log_w, mu, sigma):
    """E[dt] = (1 - pi) * sum_k w_k exp(mu_k + sigma_k^2 / 2) (legacy decode)."""
    comp = torch.exp((mu + 0.5 * sigma ** 2).clamp(max=20.0))
    return (1.0 - pi) * (log_w.exp() * comp).sum(-1)


# --------------------------------------------------------------------------- #
# Quantiles and decodes
# --------------------------------------------------------------------------- #

def quantile(q, pi, log_w, mu, sigma, iters=60):
    """Quantile of the ZILM, vectorized [N]. q: scalar or [N].

    Returns 0 where q <= pi (the atom absorbs the quantile); otherwise solves
    cont_cdf(t) = (q - pi) / (1 - pi) by bisection in log-t.
    """
    q = torch.as_tensor(q, dtype=pi.dtype, device=pi.device).expand_as(pi)
    target = ((q - pi) / (1.0 - pi).clamp(min=1e-12)).clamp(min=0.0, max=1.0 - 1e-9)
    lo = (mu - 10.0 * sigma).min(dim=-1).values                     # log-t bracket
    hi = (mu + 10.0 * sigma).max(dim=-1).values
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        go_right = cont_cdf(mid.exp(), log_w, mu, sigma) < target
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)
    t = (0.5 * (lo + hi)).exp()
    return torch.where(q <= pi, torch.zeros_like(t), t)


def eps_ball_mass(t, eps, pi, log_w, mu, sigma):
    """P(|dt - t| < eps) = (1-pi)[C(t+eps) - C(max(t-eps, 0))] + pi 1[t < eps].

    t: [N] or [N, C]; eps: scalar (days).
    """
    upper = cont_cdf(t + eps, log_w, mu, sigma)
    lower = cont_cdf((t - eps).clamp(min=0.0), log_w, mu, sigma)
    if t.dim() > pi.dim():
        pi = pi.unsqueeze(-1)
    return (1.0 - pi) * (upper - lower) + pi * (t < eps)


def eps_mode(eps, pi, log_w, mu, sigma, n_grid=49, refine=3):
    """Decode t-hat = argmax_t P(|dt - t| < eps): the JEA-optimal point estimate.

    Coarse candidates per event: 0, the K component modes exp(mu - sigma^2),
    the K component medians exp(mu), the ZILM mean and median, and an n_grid
    log-spaced grid spanning [min(mu - 3 sigma), max(mu + 3 sigma)] in log-t.
    The best candidate is then refined by `refine` rounds of local log-space
    grid search (the ball mass is sharply peaked when eps is small relative to
    the density scale, so a coarse grid alone can miss the maximizer).
    Including mean and median guarantees the decode never scores below either.
    """
    zero = torch.zeros_like(pi).unsqueeze(-1)                       # [N, 1]
    modes = torch.exp(mu - sigma ** 2)                              # [N, K]
    medians = torch.exp(mu)                                         # [N, K]
    extra = torch.stack(
        [zilm_mean(pi, log_w, mu, sigma), quantile(0.5, pi, log_w, mu, sigma)],
        dim=-1)                                                     # [N, 2]
    glo = (mu - 3.0 * sigma).min(dim=-1, keepdim=True).values       # [N, 1]
    ghi = (mu + 3.0 * sigma).max(dim=-1, keepdim=True).values
    steps = torch.linspace(0.0, 1.0, n_grid, dtype=pi.dtype, device=pi.device)
    grid = torch.exp(glo + (ghi - glo) * steps)                     # [N, G]
    cands = torch.cat([zero, modes, medians, extra, grid], dim=-1)  # [N, 3+2K+G]
    mass = eps_ball_mass(cands, eps, pi, log_w, mu, sigma)
    idx = mass.argmax(dim=-1, keepdim=True)
    best_t = cands.gather(-1, idx).squeeze(-1)                      # [N]

    spacing = (ghi - glo).squeeze(-1) / (n_grid - 1)                # [N] log-t step
    local_steps = torch.linspace(-1.0, 1.0, 17, dtype=pi.dtype, device=pi.device)
    for _ in range(refine):
        center = torch.log(best_t.clamp(min=1e-12)).unsqueeze(-1)
        local = torch.exp(center + spacing.unsqueeze(-1) * local_steps)
        # keep the incumbent so refinement can only improve; skip refining t=0
        local = torch.where(best_t.unsqueeze(-1) > 0, local, torch.zeros_like(local))
        cand2 = torch.cat([best_t.unsqueeze(-1), local], dim=-1)
        m2 = eps_ball_mass(cand2, eps, pi, log_w, mu, sigma)
        idx2 = m2.argmax(dim=-1, keepdim=True)
        best_t = cand2.gather(-1, idx2).squeeze(-1)
        spacing = spacing * (2.0 / 16.0)
    return best_t


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def pit(dt, pi, log_w, mu, sigma, generator=None):
    """Randomized PIT [N]: U(0,1) iff the model is the true law.

    Continuous observations: F(dt). Atom at 0: U * pi with U ~ Uniform(0,1)
    (randomization over the jump, Brockwell 2007).
    """
    u = torch.rand(dt.shape, dtype=dt.dtype, device=dt.device, generator=generator)
    full = pi + (1.0 - pi) * cont_cdf(dt, log_w, mu, sigma)
    return torch.where(dt == 0, u * pi, full)


def ks_uniform(p):
    """Kolmogorov-Smirnov sup-distance of samples p [N] from U(0,1)."""
    p, _ = torch.sort(p.flatten())
    n = p.numel()
    i = torch.arange(1, n + 1, dtype=p.dtype, device=p.device)
    d_plus = (i / n - p).max()
    d_minus = (p - (i - 1) / n).max()
    return float(torch.maximum(d_plus, d_minus))


def interval_coverage(dt, level, pi, log_w, mu, sigma, generator=None):
    """Empirical coverage and mean width (days) of the central `level` interval.

    Coverage is scored through the randomized PIT (cover iff PIT in
    [(1-level)/2, (1+level)/2]) rather than through deterministic quantiles:
    with an atom at 0, an equal-tailed interval whose lower endpoint hits the
    atom captures the entire tie mass and overcovers even under the true law,
    whereas the randomized PIT is exactly U(0,1) under the true law. Width is
    still the quantile-based interval length (sharpness, in days).
    """
    a = (1.0 - level) / 2.0
    p = pit(dt, pi, log_w, mu, sigma, generator=generator)
    cover = ((p >= a) & (p <= 1.0 - a)).float().mean()
    lo = quantile(a, pi, log_w, mu, sigma)
    hi = quantile(1.0 - a, pi, log_w, mu, sigma)
    return float(cover), float((hi - lo).mean())


def ece(probs, labels, n_bins=15):
    """Expected calibration error of a categorical head (top-label, equal-width bins)."""
    conf, pred = probs.max(dim=-1)
    acc = (pred == labels).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    out = torch.zeros((), device=probs.device)
    n = conf.numel()
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= edges[b]) & (conf <= edges[b + 1])
        if m.any():
            out = out + m.float().sum() / n * (acc[m].mean() - conf[m].mean()).abs()
    return float(out)


def mark_nll(logits, labels):
    """Per-event cross-entropy [N] of the mark head (labels 0-indexed)."""
    return F.cross_entropy(logits, labels, reduction="none")


# --------------------------------------------------------------------------- #
# Selective prediction
# --------------------------------------------------------------------------- #

def risk_coverage(conf, hit):
    """Risk-coverage curve from per-event confidence and 0/1 hits.

    Returns dict with coverage grid, selective risk, AURC, and accuracy at
    coverage 50% / 90% / 100%.
    """
    order = torch.argsort(conf, descending=True)
    h = hit.float()[order]
    n = h.numel()
    i = torch.arange(1, n + 1, dtype=h.dtype, device=h.device)
    sel_acc = h.cumsum(0) / i
    coverage = i / n
    risk = 1.0 - sel_acc
    aurc = float(torch.trapz(risk, coverage))
    def at(c):
        return float(sel_acc[min(max(int(math.ceil(c * n)) - 1, 0), n - 1)])
    return {
        "coverage": coverage.cpu().numpy(),
        "risk": risk.cpu().numpy(),
        "aurc": aurc,
        "acc_at_50": at(0.5),
        "acc_at_90": at(0.9),
        "acc_at_100": at(1.0),
    }


# --------------------------------------------------------------------------- #
# Self-test (CPU, synthetic): python generative_eval.py
# --------------------------------------------------------------------------- #

def _self_test():
    torch.manual_seed(0)
    N, K = 4000, 5
    pi = torch.rand(N) * 0.5                                  # tie prob in [0, .5)
    log_w = F.log_softmax(torch.randn(N, K), dim=-1)
    mu = torch.randn(N, K) * 1.5 - 1.0
    sigma = torch.rand(N, K) * 1.2 + 0.1

    # 1. quantile inverts the CDF
    for q in (0.1, 0.5, 0.9):
        t = quantile(q, pi, log_w, mu, sigma)
        nz = t > 0
        back = (pi + (1 - pi) * cont_cdf(t, log_w, mu, sigma))[nz]
        err = (back - q).abs().max()
        assert err < 1e-4, f"quantile inversion q={q}: max err {err:.2e}"
        assert torch.all(t[~nz] == 0) and torch.all(pi[~nz] >= q - 1e-9)

    # 2. eps-mode ball mass >= mass at mean and at median
    eps = 2.24e-3
    t_mode = eps_mode(eps, pi, log_w, mu, sigma)
    m_mode = eps_ball_mass(t_mode, eps, pi, log_w, mu, sigma)
    for other in (zilm_mean(pi, log_w, mu, sigma), quantile(0.5, pi, log_w, mu, sigma)):
        m_other = eps_ball_mass(other, eps, pi, log_w, mu, sigma)
        assert torch.all(m_mode >= m_other - 1e-9), "eps_mode not maximal"

    # 3. PIT of samples drawn from the same ZILM is ~U(0,1)
    g = torch.Generator().manual_seed(1)
    comp = torch.multinomial(log_w.exp(), 1, generator=g).squeeze(-1)
    z = torch.randn(N, generator=g)
    dt = torch.exp(mu[torch.arange(N), comp] + sigma[torch.arange(N), comp] * z)
    dt = torch.where(torch.rand(N, generator=g) < pi, torch.zeros_like(dt), dt)
    ks = ks_uniform(pit(dt, pi, log_w, mu, sigma, generator=g))
    assert ks < 0.025, f"PIT not uniform under the true law: KS={ks:.3f}"

    # 4. coverage of central intervals under the true law ~ nominal
    for level in (0.5, 0.9):
        cov, _ = interval_coverage(dt, level, pi, log_w, mu, sigma, generator=g)
        assert abs(cov - level) < 0.03, f"coverage {cov:.3f} != nominal {level}"

    # 5. ECE of a perfectly calibrated synthetic softmax ~ 0
    C = 6
    probs = F.softmax(torch.randn(20000, C), dim=-1)
    labels = torch.multinomial(probs, 1, generator=g).squeeze(-1)
    e = ece(probs, labels)
    assert e < 0.02, f"ECE of calibrated softmax too high: {e:.3f}"

    # 6. NLL matches Utils.mixture_time_loss conventions on the tie branch
    nll = time_nll(dt, pi, log_w, mu, sigma)
    tie = dt == 0
    assert torch.allclose(nll[tie], -torch.log(pi[tie].clamp(min=1e-12)))

    # 7. risk-coverage: confidence = true hit prob -> acc@50 >= acc@100
    hit = (torch.rand(N, generator=g) < 0.3).float()
    conf = hit * 0.8 + torch.rand(N, generator=g) * 0.2
    rc = risk_coverage(conf, hit)
    assert rc["acc_at_50"] >= rc["acc_at_100"]

    print("generative_eval self-test: all checks passed "
          f"(quantile, eps_mode, PIT KS={ks:.3f}, coverage, ECE={e:.3f}, NLL, risk-coverage)")


if __name__ == "__main__":
    _self_test()

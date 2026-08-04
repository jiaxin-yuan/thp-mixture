"""Generative-evaluation utilities for the zero-inflated LogNormal mixture (ZILM).

Parametrization matches Utils.py (mixture_time_loss / expected_dt):
    pi    = sigmoid(tie_logit)                          P(dt = 0)
    w     = softmax(time_params[..., :K])               mixture weights
    mu    = time_params[..., K:2K]                      log-space means
    sigma = softplus(time_params[..., 2K:]) + 1e-4      log-space stds
Law:  dt = 0 w.p. pi;  dt ~ sum_k w_k LogNormal(mu_k, sigma_k) w.p. 1 - pi.

All functions take flat per-event tensors: pi [N], log_w/mu/sigma [N, K],
dt [N]; candidate sets are [N, C].
"""

import math

import torch
import torch.nn.functional as F

_SQRT2 = math.sqrt(2.0)
_LOG_2PI = math.log(2.0 * math.pi)


def extract_params(tie_logit, time_params, K):
    """Head outputs -> (pi [.], log_w [., K], mu [., K], sigma [., K])."""
    pi = torch.sigmoid(tie_logit)
    log_w = F.log_softmax(time_params[..., :K], dim=-1)
    mu = time_params[..., K:2 * K]
    sigma = F.softplus(time_params[..., 2 * K:]) + 1e-4
    return pi, log_w, mu, sigma


def time_nll(dt, pi, log_w, mu, sigma):
    """Per-event NLL [N] of the ZILM: ties give a log-probability, non-ties a log-density (days^-1), as in Utils.mixture_time_loss."""
    log_pi = torch.log(pi.clamp(min=1e-12))
    log_1mpi = torch.log((1.0 - pi).clamp(min=1e-12))
    log_dt = torch.log(dt.clamp(min=1e-8)).unsqueeze(-1)            # [N, 1]
    log_pdf = (-log_dt - torch.log(sigma) - 0.5 * _LOG_2PI
               - 0.5 * ((log_dt - mu) / sigma) ** 2)                # [N, K]
    log_mix = torch.logsumexp(log_w + log_pdf, dim=-1)              # [N]
    return torch.where(dt == 0, -log_pi, -log_1mpi - log_mix)


def cont_cdf(t, log_w, mu, sigma):
    """CDF of the continuous part at t [N] or [N, C]; 0 for t <= 0."""
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
    """E[dt] = (1 - pi) * sum_k w_k exp(mu_k + sigma_k^2 / 2)."""
    comp = torch.exp((mu + 0.5 * sigma ** 2).clamp(max=20.0))
    return (1.0 - pi) * (log_w.exp() * comp).sum(-1)


def quantile(q, pi, log_w, mu, sigma, iters=60):
    """Quantile [N] of the ZILM: 0 where q <= pi, else cont_cdf(t) = (q - pi)/(1 - pi) by bisection in log-t."""
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


def pit(dt, pi, log_w, mu, sigma, generator=None):
    """Randomized PIT [N], U(0,1) iff the model is the true law; the atom at 0 is randomized over its jump (Brockwell 2007)."""
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
    """Coverage and mean width (days) of the central `level` interval.

    Coverage is scored through the randomized PIT, not deterministic quantiles:
    with an atom at 0, an interval whose lower endpoint hits the atom captures
    the whole tie mass and overcovers even under the true law. Width stays the
    quantile-based interval length.
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


def risk_coverage(conf, hit):
    """Selective-prediction curve from per-event confidence and 0/1 hits: coverage grid, risk, AURC, accuracy at 50/90/100% coverage."""
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


def _self_test():
    """CPU self-test on synthetic draws: python generative_eval.py."""
    torch.manual_seed(0)
    N, K = 4000, 5
    pi = torch.rand(N) * 0.5                                  # tie prob in [0, .5)
    log_w = F.log_softmax(torch.randn(N, K), dim=-1)
    mu = torch.randn(N, K) * 1.5 - 1.0
    sigma = torch.rand(N, K) * 1.2 + 0.1

    # quantile inverts the CDF
    for q in (0.1, 0.5, 0.9):
        t = quantile(q, pi, log_w, mu, sigma)
        nz = t > 0
        back = (pi + (1 - pi) * cont_cdf(t, log_w, mu, sigma))[nz]
        err = (back - q).abs().max()
        assert err < 1e-4, f"quantile inversion q={q}: max err {err:.2e}"
        assert torch.all(t[~nz] == 0) and torch.all(pi[~nz] >= q - 1e-9)

    # PIT of samples from the same ZILM is ~U(0,1)
    g = torch.Generator().manual_seed(1)
    comp = torch.multinomial(log_w.exp(), 1, generator=g).squeeze(-1)
    z = torch.randn(N, generator=g)
    dt = torch.exp(mu[torch.arange(N), comp] + sigma[torch.arange(N), comp] * z)
    dt = torch.where(torch.rand(N, generator=g) < pi, torch.zeros_like(dt), dt)
    ks = ks_uniform(pit(dt, pi, log_w, mu, sigma, generator=g))
    assert ks < 0.025, f"PIT not uniform under the true law: KS={ks:.3f}"

    # coverage under the true law ~ nominal
    for level in (0.5, 0.9):
        cov, _ = interval_coverage(dt, level, pi, log_w, mu, sigma, generator=g)
        assert abs(cov - level) < 0.03, f"coverage {cov:.3f} != nominal {level}"

    # ECE of a calibrated synthetic softmax ~ 0
    C = 6
    probs = F.softmax(torch.randn(20000, C), dim=-1)
    labels = torch.multinomial(probs, 1, generator=g).squeeze(-1)
    e = ece(probs, labels)
    assert e < 0.02, f"ECE of calibrated softmax too high: {e:.3f}"

    # NLL matches the tie branch of Utils.mixture_time_loss
    nll = time_nll(dt, pi, log_w, mu, sigma)
    tie = dt == 0
    assert torch.allclose(nll[tie], -torch.log(pi[tie].clamp(min=1e-12)))

    # confidence = true hit prob -> acc@50 >= acc@100
    hit = (torch.rand(N, generator=g) < 0.3).float()
    conf = hit * 0.8 + torch.rand(N, generator=g) * 0.2
    rc = risk_coverage(conf, hit)
    assert rc["acc_at_50"] >= rc["acc_at_100"]

    print("generative_eval self-test: all checks passed "
          f"(quantile, PIT KS={ks:.3f}, coverage, ECE={e:.3f}, NLL, risk-coverage)")


if __name__ == "__main__":
    _self_test()

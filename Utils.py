

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.model import get_non_pad_mask


def softplus(x, beta):
    """Softplus with the argument hard-thresholded at 20 (overflow guard)."""
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """Log-likelihood of the observed events."""
    event += math.pow(10, -9)   # guard against zero likelihood
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """Non-event log-likelihood by linear interpolation.

    Cheaper alternative to :func:`compute_integral_unbiased`, kept from the THP
    reference implementation; the training loop uses the MC estimator.
    """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """Non-event log-likelihood by Monte Carlo integration of the compensator."""

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """Event and non-event log-likelihood of each sequence in the batch."""

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)              # [B, S, num_types]
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)  # [B, S], observed type

    # event term
    event_ll = compute_event(type_lambda, non_pad_mask)     # [B, S]
    event_ll = torch.sum(event_ll, dim=-1)                  # [B]

    # non-event term, by MC integration (compute_integral_biased is the
    # cheaper linear-interpolation alternative)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)          # [B]

    return event_ll, non_event_ll



def type_loss(prediction, types):
    """Next activity cross-entropy and number of correct predictions."""

    # 1-based types back to 0-based; padding becomes -1 and is ignored
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def mixture_time_loss(tie_logit, time_params, event_time, non_pad_mask, K):
    """Zero-inflated LogNormal mixture NLL for inter-event times.

    Loss per prediction:
      if Δt == 0:  −log(π)
      else:        −log(1−π) − log Σ_k w_k · LogNormal(Δt; μ_k, σ_k)
    """
    dt = event_time[:, 1:] - event_time[:, :-1]          # [B, S-1]
    mask = non_pad_mask[:, 1:]

    pi_logit = tie_logit[:, :-1]                          # [B, S-1]
    params   = time_params[:, :-1]                        # [B, S-1, 3K]

    w_logits  = params[:, :, :K]
    mu        = params[:, :, K:2*K]
    raw_sigma = params[:, :, 2*K:]
    sigma     = F.softplus(raw_sigma) + 1e-4

    log_w    = F.log_softmax(w_logits, dim=-1)            # [B, S-1, K]
    log_pi   = F.logsigmoid(pi_logit)                     # [B, S-1]
    log_1mpi = F.logsigmoid(-pi_logit)

    is_tie = (dt == 0).float()

    # Non-tie branch: LogNormal mixture log-pdf
    log_dt  = torch.log(dt.clamp(min=1e-8)).unsqueeze(-1) # [B, S-1, 1]
    log_pdf = (
        -log_dt
        - torch.log(sigma)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * ((log_dt - mu) / sigma) ** 2
    )                                                      # [B, S-1, K]
    log_mixture = torch.logsumexp(log_w + log_pdf, dim=-1) # [B, S-1]

    nll = is_tie * (-log_pi) + (1 - is_tie) * (-log_1mpi - log_mixture)
    return (nll * mask).sum()


def expected_dt(tie_logit, time_params, non_pad_mask, K):
    """E[Δt] from the zero-inflated LogNormal mixture.

    E[Δt] = (1 − π) · Σ_k w_k · exp(μ_k + σ_k²/2)
    """
    pi = torch.sigmoid(tie_logit)

    w_logits  = time_params[:, :, :K]
    mu        = time_params[:, :, K:2*K]
    raw_sigma = time_params[:, :, 2*K:]
    sigma     = F.softplus(raw_sigma) + 1e-4

    w = F.softmax(w_logits, dim=-1)
    component_mean = torch.exp((mu + 0.5 * sigma ** 2).clamp(max=20))
    mixture_mean   = (w * component_mean).sum(dim=-1)     # [B, S]

    return (1 - pi) * mixture_mean * non_pad_mask
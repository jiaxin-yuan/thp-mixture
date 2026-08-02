

import os
import time
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import Utils
from transformer.Constants import PAD
from transformer.model import get_non_pad_mask

# Gradient-norm clip. Keeps the LogNormal mixture stable early in training,
# where E[Δt]=exp(μ+σ²/2) can otherwise diverge.
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))

# Loss weights of the standardized baseline: LL_W·ll + T_W·Δt + RT_W·rt
# (+ CE_W·ce for baseline_act). Each term is a per-event mean, so the terms
# share a scale and the defaults weight them equally.
LL_W = float(os.environ.get("LL_W", "1.0"))
T_W = float(os.environ.get("T_W", "1.0"))
RT_W = float(os.environ.get("RT_W", "1.0"))
CE_W = float(os.environ.get("CE_W", "1.0"))


def compute_baseline_time_stats(loader, device):
    """Train-split z-score stats for the baseline temporal heads.

    Returns means and stds (in days) of the next event gap Δt and of the
    remaining time, over non-pad prediction positions.
    """
    dt_sum = dt_sqsum = rt_sum = rt_sqsum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            event_time, _, r_time, event_type = (x.to(device) for x in batch)
            non_pad = get_non_pad_mask(event_type).squeeze(2)
            mask = non_pad[:, 1:]
            dt = (event_time[:, 1:] - event_time[:, :-1])[mask > 0]
            rt = r_time[:, 1:][mask > 0]
            dt_sum += dt.sum().item();  dt_sqsum += (dt * dt).sum().item()
            rt_sum += rt.sum().item();  rt_sqsum += (rt * rt).sum().item()
            n += dt.numel()
    if n == 0:
        return None
    dt_mean = dt_sum / n
    rt_mean = rt_sum / n
    dt_std = max((dt_sqsum / n - dt_mean ** 2) ** 0.5, 1e-6)
    rt_std = max((rt_sqsum / n - rt_mean ** 2) ** 0.5, 1e-6)
    return {"dt_mean": dt_mean, "dt_std": dt_std, "rt_mean": rt_mean, "rt_std": rt_std}


def _run_epoch_baseline(model, loader, optimiser=None, writer=None, epoch=0, phase="Training"):
    """One epoch of the baseline / baseline_act model (train if *optimiser*).

    baseline_act adds a categorical next-activity head whose CE joins the loss
    with weight CE_W; the mark-less baseline reports accuracy 0.
    """
    is_train = optimiser is not None
    model.train() if is_train else model.eval()
    has_mark = model.model_type == "baseline_act"

    total_ll_loss = total_time_ae = total_rt_ae = 0.0
    total_ce = total_correct = 0.0
    total_num_event = total_num_pred = 0
    start = time.time()

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch_idx, batch in enumerate(
            tqdm(loader, mininterval=2, desc=f"  - ({phase})   ", leave=False)
        ):
            event_time, time_gap, r_time, event_type = (
                x.to(next(model.parameters()).device) for x in batch
            )

            enc_out, heads = model(event_type, event_time)
            if has_mark:
                time_pred, rt_pred, mark_logit = heads
            else:
                time_pred, rt_pred = heads
            non_pad_mask = get_non_pad_mask(event_type).squeeze(2)

            # Intensity MLE term, on raw time and so unaffected by scaling.
            event_ll, non_event_ll = Utils.log_likelihood(
                model, enc_out, event_time, event_type)
            ll_loss = -torch.sum(event_ll - non_event_ll)

            # Temporal heads: the loss is computed on z-scored targets, the
            # predictions are de-standardized to days for MAE reporting.
            # time_stats=None means identity, i.e. raw days.
            st = model.time_stats
            dt_mean = dt_std = rt_mean = rt_std = None
            if st is not None:
                dt_mean, dt_std = st["dt_mean"], st["dt_std"]
                rt_mean, rt_std = st["rt_mean"], st["rt_std"]

            mask = non_pad_mask[:, 1:]
            true_dt = event_time[:, 1:] - event_time[:, :-1]
            pred_dt = time_pred.squeeze(-1)[:, :-1]
            true_rt = r_time[:, 1:]
            pred_rt = rt_pred.squeeze(-1)[:, :-1]

            if st is not None:
                t_loss = (torch.abs(pred_dt - (true_dt - dt_mean) / dt_std) * mask).sum()
                rt_loss = (torch.abs(pred_rt - (true_rt - rt_mean) / rt_std) * mask).sum()
                dt_days = pred_dt * dt_std + dt_mean
                rt_days = pred_rt * rt_std + rt_mean
            else:
                t_loss = (torch.abs(pred_dt - true_dt) * mask).sum()
                rt_loss = (torch.abs(pred_rt - true_rt) * mask).sum()
                dt_days, rt_days = pred_dt, pred_rt
            time_ae_days = (torch.abs(dt_days - true_dt) * mask).sum()
            rt_ae_days = (torch.abs(rt_days - true_rt) * mask).sum()

            ce_loss = correct = None
            if has_mark:
                ce_loss, correct = Utils.type_loss(mark_logit, event_type)

            if st is not None:
                # per-event means, so the task terms share a scale
                n_pred_b = mask.sum().clamp(min=1)
                n_ev_b = event_type.ne(PAD).sum().clamp(min=1)
                loss = (LL_W * ll_loss / n_ev_b
                        + T_W * t_loss / n_pred_b
                        + RT_W * rt_loss / n_pred_b)
                if has_mark:
                    loss = loss + CE_W * ce_loss / n_pred_b
            else:
                loss = ll_loss + t_loss + rt_loss
                if has_mark:
                    loss = loss + CE_W * ce_loss

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimiser.step()

            # MAE is always accumulated in days, for comparability across runs
            n_events = event_type.ne(PAD).sum().item()
            n_preds = n_events - event_time.shape[0]
            total_ll_loss += ll_loss.item()
            total_time_ae += time_ae_days.item()
            total_rt_ae += rt_ae_days.item()
            if has_mark:
                total_ce += ce_loss.item()
                total_correct += correct.item()
            total_num_event += n_events
            total_num_pred += n_preds

            if writer is not None and is_train:
                gs = epoch * len(loader) + batch_idx
                writer.add_scalar(f"{phase}/Batch/Total_Loss", loss.item(), gs)

    mae = total_time_ae / total_num_pred
    mae_rt = total_rt_ae / total_num_pred
    ll_avg = -total_ll_loss / total_num_event
    acc = total_correct / total_num_pred if has_mark else 0.0
    elapsed = time.time() - start

    if writer is not None:
        writer.add_scalar(f"{phase}/Epoch/MAE", mae, epoch)
        writer.add_scalar(f"{phase}/Epoch/MAE_RT", mae_rt, epoch)
        writer.add_scalar(f"{phase}/Epoch/LL", ll_avg, epoch)
        if has_mark:
            writer.add_scalar(f"{phase}/Epoch/CE", total_ce / total_num_pred, epoch)
            writer.add_scalar(f"{phase}/Epoch/Type_Accuracy", acc, epoch)
        writer.add_scalar(f"{phase}/Epoch/Time_Minutes", elapsed / 60, epoch)

    return ll_avg, mae, mae_rt, acc, elapsed


def _run_epoch_mixture(model, loader, optimiser=None, writer=None, epoch=0, phase="Training"):
    """One epoch of the mixture model (train if *optimiser* is given)."""
    is_train = optimiser is not None
    model.train() if is_train else model.eval()

    total_nll = total_ce = total_time_ae = total_rt_ae = 0.0
    total_correct = 0.0
    total_num_event = total_num_pred = 0
    start = time.time()

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch_idx, batch in enumerate(
            tqdm(loader, mininterval=2, desc=f"  - ({phase})   ", leave=False)
        ):
            event_time, time_gap, r_time, event_type = (
                x.to(next(model.parameters()).device) for x in batch
            )

            enc_out, (tie_logit, time_params, mark_logit) = model(event_type, event_time)
            non_pad_mask = get_non_pad_mask(event_type).squeeze(2)

            time_nll = Utils.mixture_time_loss(
                tie_logit, time_params, event_time, non_pad_mask, model.n_mix)
            ce_loss, correct = Utils.type_loss(mark_logit, event_type)

            loss = time_nll + ce_loss

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimiser.step()

            # Remaining time is the cumulative sum of the expected gaps
            with torch.no_grad():
                pred_dt = Utils.expected_dt(
                    tie_logit, time_params, non_pad_mask, model.n_mix)
                true_dt = event_time[:, 1:] - event_time[:, :-1]
                dt_mask = non_pad_mask[:, 1:]
                ae = (torch.abs(pred_dt[:, :-1] - true_dt) * dt_mask).sum()

                pred_rt = torch.flip(
                    torch.cumsum(torch.flip(pred_dt, [1]), dim=1), [1])
                rt_ae = (torch.abs(pred_rt[:, 1:] - r_time[:, 1:]) * dt_mask).sum()

            n_events = event_type.ne(PAD).sum().item()
            n_preds = n_events - event_time.shape[0]
            total_nll += time_nll.item()
            total_ce += ce_loss.item()
            total_correct += correct.item()
            total_time_ae += ae.item()
            total_rt_ae += rt_ae.item()
            total_num_event += n_events
            total_num_pred += n_preds

            if writer is not None and is_train:
                gs = epoch * len(loader) + batch_idx
                writer.add_scalar(f"{phase}/Batch/Total_Loss", loss.item(), gs)
                writer.add_scalar(f"{phase}/Batch/Time_NLL", time_nll.item(), gs)
                writer.add_scalar(f"{phase}/Batch/CE_Loss", ce_loss.item(), gs)

    mae = total_time_ae / total_num_pred
    mae_rt = total_rt_ae / total_num_pred
    acc = total_correct / total_num_pred
    elapsed = time.time() - start

    if writer is not None:
        writer.add_scalar(f"{phase}/Epoch/NLL", total_nll / total_num_pred, epoch)
        writer.add_scalar(f"{phase}/Epoch/CE", total_ce / total_num_pred, epoch)
        writer.add_scalar(f"{phase}/Epoch/MAE", mae, epoch)
        writer.add_scalar(f"{phase}/Epoch/MAE_RT", mae_rt, epoch)
        writer.add_scalar(f"{phase}/Epoch/Type_Accuracy", acc, epoch)
        writer.add_scalar(f"{phase}/Epoch/Time_Minutes", elapsed / 60, epoch)

    return -total_nll / total_num_pred, mae, mae_rt, acc, elapsed


def train_epoch(model, train_loader, optimiser, writer=None, epoch=0):
    """Run one training epoch (dispatches by model_type)."""
    if model.model_type in ("baseline", "baseline_act"):
        return _run_epoch_baseline(model, train_loader, optimiser, writer, epoch, "Training")
    return _run_epoch_mixture(model, train_loader, optimiser, writer, epoch, "Training")


def eval_epoch(model, val_loader, writer=None, epoch=0, phase="Validation"):
    """Run one evaluation epoch (dispatches by model_type)."""
    if model.model_type in ("baseline", "baseline_act"):
        return _run_epoch_baseline(model, val_loader, None, writer, epoch, phase)
    return _run_epoch_mixture(model, val_loader, None, writer, epoch, phase)


def train_model(
    model,
    n_epochs: int,
    train_loader,
    val_loader,
    optimiser,
    scheduler,
    model_save_path: str,
    filename: str,
    lr: float,
    batch_size: int,
    results_dir: str = "results",
    log_dir: str = "runs",
    patience: int = 0,
):
    """Train for up to *n_epochs*, with checkpointing and TensorBoard logging.

    Three checkpoints are kept independently, each on the best validation value
    of its metric:
    - ``*_best_mae.pth``    next event time MAE
    - ``*_best_mae_rt.pth`` remaining time MAE
    - ``*_best_ll.pth``     event log-likelihood

    Training stops once none of the three improves for *patience* consecutive
    epochs; patience=0 disables early stopping. Per-epoch validation metrics
    are appended to ``<results_dir>/raw_<filename>.txt``.
    """
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{filename}_{timestamp}"
    writer          = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))

    print(f"\n{'='*80}")
    print(f"TensorBoard → {os.path.join(log_dir, experiment_name)}")
    print(f"Run: tensorboard --logdir={log_dir}")
    print(f"{'='*80}\n")

    base = os.path.splitext(model_save_path)[0]
    save_paths = {
        "mae":            f"{base}_best_mae.pth",
        "mae_rt":         f"{base}_best_mae_rt.pth",
        "log_likelihood": f"{base}_best_ll.pth",
    }

    best = {
        "mae":            {"value": float("inf"),  "epoch": 0, "minimize": True},
        "mae_rt":         {"value": float("inf"),  "epoch": 0, "minimize": True},
        "log_likelihood": {"value": float("-inf"), "epoch": 0, "minimize": False},
    }

    print("Checkpoint paths:")
    for k, p in save_paths.items():
        print(f"  {k:20s} → {p}")
    print()

    raw_log_path = os.path.join(results_dir, f"raw_{filename}.txt")
    epoch_times  = []
    total_start  = time.time()
    epochs_no_improve = 0
    stopped_early     = False

    for i in range(n_epochs):
        print(f"[Epoch {i+1}/{n_epochs}]")
        epoch_start = time.time()

        # ---- train ----
        tr_ll, tr_mae, tr_rt, tr_acc, tr_t = train_epoch(model, train_loader, optimiser, writer, i)
        print(
            f"  Train  ll={tr_ll:8.5f}  MAE={tr_mae:8.5f}  "
            f"MAE_rt={tr_rt:8.5f}  acc={tr_acc:.4f}  t={tr_t/60:.2f}min"
        )

        # ---- validate ----
        vl_ll, vl_mae, vl_rt, vl_acc, vl_t = eval_epoch(model, val_loader, writer, i)
        print(
            f"  Val    ll={vl_ll:8.5f}  MAE={vl_mae:8.5f}  "
            f"MAE_rt={vl_rt:8.5f}  acc={vl_acc:.4f}  t={vl_t/60:.2f}min"
        )

        epoch_total = time.time() - epoch_start
        epoch_times.append(epoch_total)

        # ---- raw log ----
        with open(raw_log_path, "a") as f:
            f.write(f"{i}, {vl_ll:8.5f}, {vl_mae:8.5f}, {vl_rt:8.5f}, {vl_acc:.5f}\n")

        # ---- checkpointing ----
        current = {"mae": vl_mae, "mae_rt": vl_rt, "log_likelihood": vl_ll}
        checkpoint = {
            "epoch":               i,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_mae":             vl_mae,
            "val_mae_rt":          vl_rt,
            "val_log_likelihood":  vl_ll,
            "time_stats":          model.time_stats,
        }

        saved = []
        for metric, info in best.items():
            val = current[metric]
            better = (val < info["value"]) if info["minimize"] else (val > info["value"])
            if better:
                best[metric]["value"] = val
                best[metric]["epoch"] = i + 1
                torch.save(checkpoint, save_paths[metric])
                writer.add_text(f"Model_Save/{metric}", f"Saved at epoch {i+1}", i)
                saved.append(metric)

        if saved:
            print(f"  Saved: {', '.join(saved).upper()}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---- TensorBoard scalars ----
        writer.add_scalar("Train/Learning_Rate",     optimiser.param_groups[0]["lr"], i)
        writer.add_scalar("Best/MAE",                best["mae"]["value"],            i)
        writer.add_scalar("Best/MAE_RT",             best["mae_rt"]["value"],         i)
        writer.add_scalar("Best/Log_Likelihood",     best["log_likelihood"]["value"], i)
        writer.add_scalar("Time/Epoch_Total_Minutes", epoch_total / 60,               i)
        writer.add_scalar("Time/Cumulative_Hours",    sum(epoch_times) / 3600,        i)
        if len(epoch_times) > 1:
            writer.add_scalar("Time/Average_Epoch_Minutes", np.mean(epoch_times) / 60, i)

        scheduler.step()

        # ---- early stopping ----
        if patience > 0 and epochs_no_improve >= patience:
            print(
                f"\n[Early stop] No val improvement for {patience} epochs "
                f"(stopped at epoch {i+1}/{n_epochs})."
            )
            stopped_early = True
            break

    # ---- final summary ----
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE" + (" (early-stopped)" if stopped_early else ""))
    for metric, info in best.items():
        print(f"  {metric:20s}  {info['value']:.5f}  (epoch {info['epoch']})")
    print(f"  Total time : {total_time/3600:.2f}h  |  Avg/epoch: {np.mean(epoch_times)/60:.2f}min")
    print(f"{'='*80}")

    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "epochs": n_epochs},
        {
            "best_val_mae":     best["mae"]["value"],
            "best_val_mae_rt":  best["mae_rt"]["value"],
            "best_val_ll":      best["log_likelihood"]["value"],
            "total_train_hours": total_time / 3600,
            "avg_epoch_min":    np.mean(epoch_times) / 60,
        },
    )
    writer.close()
    print(f"\nTensorBoard logs: {os.path.join(log_dir, experiment_name)}")
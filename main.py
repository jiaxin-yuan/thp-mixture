
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim

from preprocess.dataset import df_to_dict, get_dataloader
from trainer.train import train_model, eval_epoch, compute_baseline_time_stats
from utils.reproducibility import set_all_seeds
from transformer.model import Transformer


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_dataset", required=True,
                        help="<dir>/<stem>.csv, from which <dir>/{train,val,test}_"
                             "<stem>.csv are read")

    parser.add_argument("--batch_size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--epoch",      type=int,   default=1,  help="Maximum number of epochs")
    parser.add_argument("--patience",   type=int,   default=0,
                        help="Epochs without validation improvement before "
                             "early stopping; 0 disables it")
    parser.add_argument("--lr",         type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    parser.add_argument("--n_mix",      type=int,   default=5,    help="LogNormal mixture components")
    parser.add_argument("--model_type", default="mixture",
                        choices=["baseline_act", "mixture"],
                        help="Head: 'mixture' = THP-M (LogNormal mixture + next "
                             "activity), 'baseline_act' = THP-B (intensity MLE "
                             "time heads + next activity)")

    parser.add_argument("--device",  default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--log_dir", default="runs", help="TensorBoard log directory")

    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--test",  action="store_true", help="Run test evaluation")
    parser.add_argument("--model_path", default=None,
                        help="Checkpoint path for --test (default: best MAE checkpoint)")

    args = parser.parse_args()

    if not (args.train or args.test):
        parser.error("At least one of --train or --test is required.")

    return args


def setup_directories(log_dir: str) -> None:
    """Create the output directories if they do not exist."""
    os.makedirs("results",      exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs(log_dir,        exist_ok=True)


def init_result_file(output_file: str) -> None:
    """Write a timestamped header to the per-run result text file."""
    current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
    with open(output_file, "w") as f:
        f.write(f"Starting time: {current_time}\n")


def resolve_device(requested: str) -> torch.device:
    """Return a torch.device, falling back to CPU if CUDA is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_data(directory: str, fold_filename: str, batch_size: int):
    """Load the three splits as (train, val, test) loaders plus num_types."""
    print("Loading and preprocessing data ...")
    train_out, val_out, test_out = df_to_dict(
        directory=directory,
        fold_filename=fold_filename,
    )
    num_types = train_out["dim_process"]

    print(f"  Activity types : {num_types}")
    print(f"  Train cases    : {len(train_out['train'])}")
    print(f"  Val   cases    : {len(val_out['val'])}")
    print(f"  Test  cases    : {len(test_out['test'])}")

    train_loader = get_dataloader(train_out["train"], batch_size, shuffle=True)
    val_loader   = get_dataloader(val_out["val"],     batch_size, shuffle=False)
    test_loader  = get_dataloader(test_out["test"],   batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_types


def build_model(num_types: int, device: torch.device, n_mix: int = 5,
                model_type: str = "mixture") -> Transformer:
    """Instantiate the Transformer with the paper hyper-parameters."""
    model = Transformer(
        num_types=num_types,
        d_model=36,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        n_mix=n_mix,
        model_type=model_type,
    )
    model.to(device)
    return model


def run_training(
    model: Transformer,
    train_loader,
    val_loader,
    model_save_path: str,
    fold_filename: str,
    args: argparse.Namespace,
) -> None:
    """Set up optimiser / scheduler and run the training loop."""
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    # THP-B regresses z-scored targets; the mixture head works in log-space
    if model.model_type == "baseline_act":
        device = next(model.parameters()).device
        model.time_stats = compute_baseline_time_stats(train_loader, device)
        print(f"  Time standardization (train z-score): {model.time_stats}")

    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-5,
    )
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

    train_model(
        model=model,
        n_epochs=args.epoch,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=optimiser,
        scheduler=scheduler,
        model_save_path=model_save_path,
        filename=fold_filename,
        lr=args.lr,
        batch_size=args.batch_size,
        results_dir="results",
        log_dir=args.log_dir,
        patience=args.patience,
    )

    print("Training completed!")
    print(f"\nTo view TensorBoard:  tensorboard --logdir={args.log_dir}")
    print("Then open: http://localhost:6006")


def run_testing(
    model: Transformer,
    test_loader,
    model_save_path: str,
    output_file: str,
    args: argparse.Namespace,
) -> None:
    """Evaluate the best-MAE checkpoint, or --model_path, on the test split."""
    print("\n" + "=" * 80)
    print("STARTING TESTING")
    print("=" * 80)

    if args.model_path:
        ckpt_path = args.model_path
    else:
        base      = os.path.splitext(model_save_path)[0]
        ckpt_path = f"{base}_best_mae.pth"

    if not os.path.isfile(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.time_stats = checkpoint.get("time_stats")
    print(f"Loaded checkpoint (epoch {checkpoint['epoch'] + 1}): {ckpt_path}")
    if model.time_stats is not None:
        print(f"  Using time standardization: {model.time_stats}")

    test_ll, test_mae, test_mae_rt, test_acc, elapsed = eval_epoch(
        model, test_loader, phase="Test"
    )

    print(f"\nTest results:")
    print(f"  Log-likelihood      : {test_ll:.5f}")
    print(f"  MAE  (next event)   : {test_mae:.5f}  days")
    print(f"  MAE  (remaining)    : {test_mae_rt:.5f}  days")
    print(f"  Type accuracy       : {test_acc:.5f}")
    print(f"  Elapsed             : {elapsed / 60:.2f} min")

    with open(output_file, "a") as f:
        f.write(
            f"TEST  ll={test_ll:.5f}  mae={test_mae:.5f}  mae_rt={test_mae_rt:.5f}  acc={test_acc:.5f}\n"
        )


def main() -> None:
    """Parse arguments, load data, build the model, and train / test."""
    args          = parse_args()
    set_all_seeds(args.seed)

    directory     = str(Path(args.fold_dataset).parent)
    fold_filename = Path(args.fold_dataset).stem
    output_file   = os.path.join("results", f"{fold_filename}_{args.model_type}.txt")

    setup_directories(args.log_dir)
    init_result_file(output_file)

    device = resolve_device(args.device)

    train_loader, val_loader, test_loader, num_types = load_data(
        directory, fold_filename, args.batch_size
    )

    model           = build_model(num_types, device, n_mix=args.n_mix,
                                     model_type=args.model_type)
    model_save_path = os.path.join("saved_models", f"{fold_filename}_{args.model_type}_best_model.pth")

    if args.train:
        run_training(model, train_loader, val_loader,
                     model_save_path, fold_filename, args)

    if args.test:
        run_testing(model, test_loader, model_save_path, output_file, args)

    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
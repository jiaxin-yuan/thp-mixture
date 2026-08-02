
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pathlib import Path
from typing import Dict, List, Tuple

from transformer.Constants import PAD

# Raw CSV -> per-case event sequences for the DataLoaders.

CASE_COL  = "CaseID"
TIME_COL  = "Timestamp"
ACT_COL   = "Activity"
RT_COL    = "remaining_time"

SECONDS_PER_DAY = 86_400.0


def load_dataframes(
    directory: str,
    fold_filename: str,
    extension: str = ".csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ``{train,val,test}_<fold_filename>.csv`` from *directory*.

    Returns ``(train_df, val_df, test_df)`` with ``Timestamp`` already parsed
    to ``datetime64``.
    """
    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(directory, name + extension)
        df = pd.read_csv(path)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        return df

    train_df     = _read(f"train_{fold_filename}")
    val_df       = _read(f"val_{fold_filename}")
    test_df      = _read(f"test_{fold_filename}")
    return train_df, val_df, test_df


def build_event_type_map(train_df: pd.DataFrame) -> Tuple[Dict[str, int], int]:
    """Map activity names to zero-based indices, in sorted order.

    Built from the train split only, so no activity vocabulary leaks from
    val/test. Returns ``(event_types, dim_process)``.
    """
    unique_acts = np.unique(train_df[ACT_COL])
    event_types = {name: idx for idx, name in enumerate(unique_acts)}
    return event_types, unique_acts.size


def add_time_features(df: pd.DataFrame, event_types: Dict[str, int]) -> pd.DataFrame:
    """Add the time and activity features to *df* in-place.

    New columns, all in days (float32) except the last:
    ``time_since_start``, ``time_since_last_event``, ``remaining_time``,
    ``type_event`` (integer activity code).

    A ``CaseEndTime`` column, written by prefix extraction, defines the case
    end for ``remaining_time``; without it the last event of the prefix is
    used, which underestimates the remaining time on truncated prefixes.
    """
    g = df.groupby(CASE_COL, sort=False)

    df["time_since_start"] = (
        (df[TIME_COL] - g[TIME_COL].transform("min"))
        .dt.total_seconds()
        .div(SECONDS_PER_DAY)
        .astype("float32")
    )
    df["time_since_last_event"] = (
        (df[TIME_COL] - g[TIME_COL].shift(1))
        .fillna(pd.Timedelta(0))
        .dt.total_seconds()
        .div(SECONDS_PER_DAY)
        .astype("float32")
    )

    if "CaseEndTime" in df.columns:
        case_end = pd.to_datetime(df["CaseEndTime"])
        df[RT_COL] = (
            (case_end - df[TIME_COL])
            .dt.total_seconds()
            .div(SECONDS_PER_DAY)
            .astype("float32")
        )
    else:
        df[RT_COL] = (
            (g[TIME_COL].transform("max") - df[TIME_COL])
            .dt.total_seconds()
            .div(SECONDS_PER_DAY)
            .astype("float32")
        )

    act_codes = df[ACT_COL].map(event_types)
    n_unknown = act_codes.isna().sum()
    if n_unknown > 0:
        print(f"  [Warning] {n_unknown} events with unseen activity mapped to UNK")
        # Dedicated UNK index (max_known + 1): keeps unseen activities clear of
        # PAD=0 after the +1 shift applied in EventData.
        unk_idx = len(event_types)
        act_codes = act_codes.fillna(unk_idx)
    df["type_event"] = act_codes.astype(int)

    return df


def df_to_sequences(df: pd.DataFrame) -> List[List[Dict]]:
    """Convert a feature-enriched DataFrame into one event list per case."""
    data = []
    for _, grp in df.groupby(CASE_COL, sort=False):
        seq = [
            {
                "time_since_start":      float(t0),
                "time_since_last_event": float(dt),
                "remaining_time":        float(rt),
                "type_event":            int(act),
            }
            for t0, dt, rt, act in zip(
                grp["time_since_start"].values,
                grp["time_since_last_event"].values,
                grp["remaining_time"].values,
                grp["type_event"].values,
            )
        ]
        data.append(seq)
    return data


def df_to_dict(
    directory: str,
    fold_filename: str,
    extension: str = ".csv",
) -> Tuple[dict, dict, dict]:
    """CSV splits -> event sequences ready for the DataLoaders.

    Each of ``(train_out, val_out, test_out)`` holds ``dim_process``,
    ``max_length``, and its split data under ``"train"`` / ``"val"`` / ``"test"``.
    """
    train_df, val_df, test_df = load_dataframes(
        directory, fold_filename, extension
    )
    event_types, dim_process = build_event_type_map(train_df)

    max_len = 0
    has_unk = False
    for df in (train_df, val_df, test_df):
        add_time_features(df, event_types)
        if (df["type_event"] == len(event_types)).any():
            has_unk = True
        case_max = df.groupby(CASE_COL, sort=False).size().max()
        if case_max > max_len:
            max_len = case_max
    if has_unk:
        dim_process += 1  # room for the UNK type in the embedding

    train_seq = df_to_sequences(train_df)
    val_seq   = df_to_sequences(val_df)
    test_seq  = df_to_sequences(test_df)

    common = {"dim_process": dim_process, "max_length": max_len}
    return (
        {**common, "train": train_seq},
        {**common, "val":   val_seq},
        {**common, "test":  test_seq},
    )



class EventData(torch.utils.data.Dataset):
    """Dataset over the event sequences produced by :func:`df_to_sequences`."""

    def __init__(self, data: List[List[Dict]]) -> None:
        self.time     = [[e["time_since_start"]      for e in inst] for inst in data]
        self.time_gap = [[e["time_since_last_event"]  for e in inst] for inst in data]
        self.r_time   = [[e["remaining_time"]          for e in inst] for inst in data]
        # Activity indices are 1-based inside the model (0 is reserved for PAD)
        self.activity = [[e["type_event"] + 1          for e in inst] for inst in data]
        self.length   = len(data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return (
            self.time[idx],
            self.time_gap[idx],
            self.r_time[idx],
            self.activity[idx],
        )


def _pad_time(insts: List[List[float]]) -> torch.Tensor:
    """Right-pad a list of float sequences to the maximum length in *insts*."""
    max_len = max(len(s) for s in insts)
    batch   = np.array([s + [PAD] * (max_len - len(s)) for s in insts])
    return torch.tensor(batch, dtype=torch.float32)


def _pad_type(insts: List[List[int]]) -> torch.Tensor:
    """Right-pad a list of integer sequences to the maximum length in *insts*."""
    max_len = max(len(s) for s in insts)
    batch   = np.array([s + [PAD] * (max_len - len(s)) for s in insts])
    return torch.tensor(batch, dtype=torch.long)


def collate_fn(insts):
    """Pad the variable-length sequences of a batch."""
    time, time_gap, rt, activity = zip(*insts)
    return (
        _pad_time(time),
        _pad_time(time_gap),
        _pad_time(rt),
        _pad_type(activity),
    )


def get_dataloader(
    data: List[List[Dict]],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    """Wrap the event sequences in a padded DataLoader."""
    dataset = EventData(data)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
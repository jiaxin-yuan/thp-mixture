
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from typing import Dict, List, Tuple

from transformer.Constants import PAD

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
    """Read the three ``{train,val,test}_<fold_filename>.csv`` splits."""
    def _read(name: str) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(directory, name + extension))
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        return df

    return (_read(f"train_{fold_filename}"),
            _read(f"val_{fold_filename}"),
            _read(f"test_{fold_filename}"))


def build_event_type_map(train_df: pd.DataFrame) -> Tuple[Dict[str, int], int]:
    """Activity names to zero-based indices, from the train split only."""
    unique_acts = np.unique(train_df[ACT_COL])
    return {name: idx for idx, name in enumerate(unique_acts)}, unique_acts.size


def add_time_features(df: pd.DataFrame, event_types: Dict[str, int]) -> pd.DataFrame:
    """Add time_since_start, time_since_last_event, remaining_time (days) and type_event, the case end coming from CaseEndTime when present."""
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

    # Without CaseEndTime the prefix's last event stands in, which
    # underestimates the remaining time on truncated prefixes.
    end = (pd.to_datetime(df["CaseEndTime"]) if "CaseEndTime" in df.columns
           else g[TIME_COL].transform("max"))
    df[RT_COL] = ((end - df[TIME_COL]).dt.total_seconds()
                  .div(SECONDS_PER_DAY).astype("float32"))

    act_codes = df[ACT_COL].map(event_types)
    n_unknown = act_codes.isna().sum()
    if n_unknown > 0:
        print(f"  [Warning] {n_unknown} events with unseen activity mapped to UNK")
        act_codes = act_codes.fillna(len(event_types))    # UNK = max_known + 1
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
    """CSV splits to event sequences, each dict holding dim_process, max_length and its own split."""
    train_df, val_df, test_df = load_dataframes(directory, fold_filename, extension)
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
        dim_process += 1

    common = {"dim_process": dim_process, "max_length": max_len}
    return (
        {**common, "train": df_to_sequences(train_df)},
        {**common, "val":   df_to_sequences(val_df)},
        {**common, "test":  df_to_sequences(test_df)},
    )


class EventData(torch.utils.data.Dataset):
    """Dataset over the event sequences produced by df_to_sequences."""

    def __init__(self, data: List[List[Dict]]) -> None:
        self.time     = [[e["time_since_start"]      for e in inst] for inst in data]
        self.time_gap = [[e["time_since_last_event"] for e in inst] for inst in data]
        self.r_time   = [[e["remaining_time"]        for e in inst] for inst in data]
        # 1-based inside the model, 0 is PAD
        self.activity = [[e["type_event"] + 1        for e in inst] for inst in data]
        self.length   = len(data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return (self.time[idx], self.time_gap[idx],
                self.r_time[idx], self.activity[idx])


def _pad_time(insts: List[List[float]]) -> torch.Tensor:
    """Right-pad float sequences to the batch maximum."""
    max_len = max(len(s) for s in insts)
    return torch.tensor(np.array([s + [PAD] * (max_len - len(s)) for s in insts]),
                        dtype=torch.float32)


def _pad_type(insts: List[List[int]]) -> torch.Tensor:
    """Right-pad integer sequences to the batch maximum."""
    max_len = max(len(s) for s in insts)
    return torch.tensor(np.array([s + [PAD] * (max_len - len(s)) for s in insts]),
                        dtype=torch.long)


def collate_fn(insts):
    """Pad the variable-length sequences of a batch."""
    time, time_gap, rt, activity = zip(*insts)
    return _pad_time(time), _pad_time(time_gap), _pad_time(rt), _pad_type(activity)


def get_dataloader(
    data: List[List[Dict]],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    """Wrap the event sequences in a padded DataLoader."""
    return torch.utils.data.DataLoader(
        EventData(data),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

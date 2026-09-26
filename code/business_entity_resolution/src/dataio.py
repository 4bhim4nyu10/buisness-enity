"""Loading / writing the challenge's TSV files. Nothing clever, but every
read and write goes through here so the tab-separator rule can never be
violated by accident."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

SOURCE_FILES = {1: "source1.tsv", 2: "source2.tsv", 3: "source3.tsv"}
REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]


def _read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
        engine="python",
    )
    df.columns = [c.strip() for c in df.columns]
    if len(df.columns) == 1:
        raise ValueError(
            f"{path} parsed into a single column — the file is probably not "
            "tab-separated, or the separator was lost. Refusing to continue."
        )
    return df


@dataclass
class SourceSet:
    """The three source tables of one split (train or test)."""

    split: str
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame

    def all_records(self) -> pd.DataFrame:
        return pd.concat([self.s1, self.s2, self.s3], ignore_index=True)

    def target(self, source: int) -> pd.DataFrame:
        return self.s2 if source == 2 else self.s3


def load_sources(data_dir: str, split: str) -> SourceSet:
    """data_dir/split/{split}_source{1,2,3}.tsv"""
    frames = {}
    for k, _ in SOURCE_FILES.items():
        path = os.path.join(data_dir, split, f"{split}_source{k}.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = _read_tsv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}; found {list(df.columns)}")
        frames[k] = df[REQUIRED_COLS].copy()
    return SourceSet(split=split, s1=frames[1], s2=frames[2], s3=frames[3])


def load_ground_truth(data_dir: str, split: str = "train") -> dict[str, set[str]]:
    path = os.path.join(data_dir, split, f"{split}_ground_truth.tsv")
    df = _read_tsv(path)
    out: dict[str, set[str]] = {}
    for sid, ids in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        ids = (ids or "").strip()
        out[sid] = {x.strip() for x in ids.split(",") if x.strip()} if ids else set()
    return out


def write_id_list_tsv(
    path: str,
    rows: Iterable[tuple[str, Sequence[str]]],
    id_column: str,
) -> None:
    """Write `source1_entity_id \\t <comma list>`.

    Deduplicates each list, preserves given order, never quotes, always writes a
    row even when the list is empty.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{id_column}\n")
        for sid, ids in rows:
            seen: set[str] = set()
            ordered = []
            for i in ids:
                if i and i not in seen:
                    seen.add(i)
                    ordered.append(i)
            fh.write(f"{sid}\t{','.join(ordered)}\n")


def write_matching_results(path: str, rows) -> None:
    write_id_list_tsv(path, rows, "matched_entity_ids")


def write_candidate_pairs(path: str, rows) -> None:
    write_id_list_tsv(path, rows, "candidate_entity_ids")

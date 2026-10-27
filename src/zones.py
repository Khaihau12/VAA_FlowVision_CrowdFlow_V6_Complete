from __future__ import annotations
import numpy as np
import pandas as pd

ZONE_ORDER = ["Zone A", "Zone B", "Zone C"]


def fit_pca_zones(gt: pd.DataFrame) -> dict:
    """Fit the generic 3-zone PCA partition on development + validation GT only.

    This mirrors the final study protocol: TEST is never used to fit the zone partition.
    The sign of a PCA component is arbitrary, so zone labels are oriented to reproduce
    the protocol convention used in the submitted result tables (A=high PCA score,
    B=middle, C=low).
    """
    fit = gt[gt["split"].isin(["train", "val"])][["ground_x", "ground_y"]].dropna().to_numpy(float)
    if len(fit) < 3:
        raise ValueError("Not enough development/validation ground-truth points to fit zones.")
    mean = fit.mean(axis=0)
    _, _, vh = np.linalg.svd(fit - mean, full_matrices=False)
    pc1 = vh[0]
    scores = (fit - mean) @ pc1
    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    return {"mean": mean, "pc1": pc1, "q1": float(q1), "q2": float(q2)}


def assign_zones(df: pd.DataFrame, model: dict) -> pd.Series:
    xy = df[["ground_x", "ground_y"]].to_numpy(float)
    scores = (xy - model["mean"]) @ model["pc1"]
    # PCA sign is arbitrary. This orientation matches the final WILDTRACK tables:
    # high score -> Zone A, middle -> B, low -> C.
    labels = np.where(scores < model["q1"], "Zone C", np.where(scores < model["q2"], "Zone B", "Zone A"))
    return pd.Series(labels, index=df.index, name="zone")


def occupancy_table(df: pd.DataFrame, model: dict, index_col: str = "gt_index") -> pd.DataFrame:
    temp = df.copy()
    temp["zone"] = assign_zones(temp, model)
    out = temp.groupby([index_col, "zone"]).size().unstack(fill_value=0)
    return out.reindex(columns=ZONE_ORDER, fill_value=0).sort_index()


def zone_centroids(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    temp = df.copy()
    temp["zone"] = assign_zones(temp, model)
    return temp.groupby("zone")[["ground_x", "ground_y"]].mean().reindex(ZONE_ORDER)

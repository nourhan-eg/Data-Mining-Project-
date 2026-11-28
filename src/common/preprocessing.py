"""
preprocessing.py - Member 1 (common code)

Shared preprocessing helpers for the whole team.

LEAKAGE CONTRACT (from the team split, frozen Day 1):
  - remove_duplicates is STATELESS: safe to call on the full dataset
    BEFORE the train/test split.
  - handle_missing, handle_outliers and scale_features are FITTED
    functions: every statistic they use (median/mode, IQR bounds,
    mean/std or min/max) is computed from the DataFrame they are
    handed. They must therefore be called on the TRAINING data only
    (inside members 2/4's fit_preprocess). Never on the full data,
    never on the test set.

  To transform the test set (or new rows) afterwards, reuse the
  statistics the fitted call recorded on the returned frame:
      train_df = handle_missing(train_df)   # fit happens here
      stats = train_df.attrs["fitted_stats"]
      test_df  = handle_missing(test_df, stats=stats)   # transform only
  The `stats` argument is optional; omitting it keeps the exact
  prototype in the team split and fits on the passed data.
"""

import numpy as np
import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and print how many were removed (safe before split)"""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[remove_duplicates] removed {removed} duplicate rows ({before} -> {len(df)})")
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "auto",
                   stats: dict = None) -> pd.DataFrame:
    """numeric -> median, categorical -> mode. FITTED stat: call on train only"""
    df = df.copy()
    fitted = {} if stats is None else dict(stats)
    fill_values = {}
    for col in df.columns:
        if not df[col].isna().any():
            continue
        if col in fitted:
            value = fitted[col]
        elif strategy == "auto":
            if pd.api.types.is_numeric_dtype(df[col]):
                value = df[col].median()
            else:
                mode = df[col].mode(dropna=True)
                value = mode.iloc[0] if len(mode) else ""
        else:
            value = df[col].median() if strategy == "median" else df[col].mode(dropna=True).iloc[0]
        df[col] = df[col].fillna(value)
        fitted[col] = value
        fill_values[col] = value
    if stats is None:
        df.attrs["fitted_stats"] = fitted
        print(f"[handle_missing] fitted & filled {len(fill_values)} column(s): {fill_values}")
    else:
        print(f"[handle_missing] transformed with given stats on {len(fill_values)} column(s)")
    return df


def handle_outliers(df: pd.DataFrame, columns: list, method: str = "iqr",
                    stats: dict = None) -> pd.DataFrame:
    """IQR capping (cap, not drop). FITTED stat: call on train only"""
    df = df.copy()
    fitted = {} if stats is None else dict(stats)
    capped_report = {}
    for col in columns:
        if col in fitted:
            low, high = fitted[col]
        elif method == "iqr":
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            raise ValueError(f"unsupported outlier method: {method!r} (only 'iqr')")
        n_capped = int(((df[col] < low) | (df[col] > high)).sum())
        df[col] = df[col].clip(lower=low, upper=high)
        fitted[col] = (low, high)
        capped_report[col] = n_capped
    if stats is None:
        df.attrs["fitted_stats"] = fitted
        print(f"[handle_outliers] fitted IQR caps; capped values per column: {capped_report}")
    else:
        print(f"[handle_outliers] transformed with given caps on {len(capped_report)} column(s)")
    return df


def scale_features(df: pd.DataFrame, columns: list, method: str = "standard",
                   stats: dict = None) -> pd.DataFrame:
    """standard or minmax scaling. FITTED stat: call on train only"""
    df = df.copy()
    fitted = {} if stats is None else dict(stats)
    for col in columns:
        if col in fitted:
            params = fitted[col]
        elif method == "standard":
            params = (float(df[col].mean()), float(df[col].std(ddof=1)))
        elif method == "minmax":
            params = (float(df[col].min()), float(df[col].max()))
        else:
            raise ValueError(f"unsupported scaling method: {method!r} (use 'standard' or 'minmax')")
        a, b = params
        if method == "standard":
            span = b if b != 0 else 1.0        # b = std
        else:  # minmax
            span = (b - a) if (b - a) != 0 else 1.0  # b = max, a = min
        df[col] = (df[col] - a) / span
        fitted[col] = params
    if stats is None:
        df.attrs["fitted_stats"] = fitted
        print(f"[scale_features] fitted '{method}' scaling on {len(columns)} column(s)")
    else:
        print(f"[scale_features] transformed with given '{method}' stats on {len(columns)} column(s)")
    return df

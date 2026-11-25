"""
data_loader.py - Member 1 (common code)

Loaders for the three raw datasets. Defaults come from config.py;
no paths or column names are hardcoded anywhere else.
"""

import pandas as pd

import config


def load_nsl_kdd(path: str = config.RAW_NSL_KDD) -> pd.DataFrame:
    """Load raw NSL-KDD (agreed ~10K-row sample)"""
    df = pd.read_csv(path)
    print(f"[load_nsl_kdd] loaded {df.shape[0]} rows x {df.shape[1]} cols from {path}")
    return df


def load_student(path: str = config.RAW_STUDENT) -> pd.DataFrame:
    """Load Student Performance (10K rows)"""
    df = pd.read_csv(path)
    print(f"[load_student] loaded {df.shape[0]} rows x {df.shape[1]} cols from {path}")
    return df


def load_online_retail(path: str = config.RAW_RETAIL) -> pd.DataFrame:
    """Load raw Online Retail"""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    print(f"[load_online_retail] loaded {df.shape[0]} rows x {df.shape[1]} cols from {path}")
    return df

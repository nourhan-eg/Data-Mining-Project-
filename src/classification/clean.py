"""
clean.py - Member 1 (Classification — Data Cleaning)

Stateless cleaning of the NSL-KDD dataset.

WHAT THIS FILE DOES (and what it does NOT do):
  - remove_duplicates  (safe before split, stateless)
  - fix dtypes          (stateless)
  - separate the target column ('label') from the features
  - save the cleaned version to data/processed/nsl_kdd_clean.csv
  - report column types so member 2 can build the preprocessor

  NO encoding, scaling, feature selection, imputation or resampling.
  Those are FITTED operations and belong to member 2 (fit_preprocess).
"""

import pandas as pd

import config
from src.common.preprocessing import remove_duplicates


def clean_nsl_kdd(df: pd.DataFrame) -> tuple:
    """Stateless cleaning only (remove_duplicates, dtypes, split off target).
    Returns (X_clean, y) and saves data/processed/nsl_kdd_clean.csv.
    NO encoding / scaling / selection / resampling here."""

    print(f"[clean_nsl_kdd] raw shape: {df.shape}")

    # 1. Remove duplicates (stateless — safe before split)
    df = remove_duplicates(df)

    # 2. Fix dtypes — make sure numeric columns are numeric, etc.
    #    NSL-KDD has 3 categorical (str) columns and the rest numeric.
    #    'difficulty_level' is metadata, not a feature — drop it.
    if "difficulty_level" in df.columns:
        df = df.drop(columns=["difficulty_level"])
        print("[clean_nsl_kdd] dropped 'difficulty_level' (metadata, not a feature)")

    # Ensure numeric columns are truly numeric (cast in case of mixed types)
    cat_cols = ["protocol_type", "service", "flag"]
    for col in df.columns:
        if col in cat_cols or col == config.NSL_KDD_TARGET:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Separate target from features
    target_col = config.NSL_KDD_TARGET
    y = df[target_col].copy()
    X_clean = df.drop(columns=[target_col])

    print(f"[clean_nsl_kdd] after cleaning: X {X_clean.shape}, y {y.shape}")
    print(f"[clean_nsl_kdd] class distribution:\n{y.value_counts()}")

    # 4. Save the cleaned data (features + target together for downstream)
    clean_df = pd.concat([X_clean, y], axis=1)
    clean_df.to_csv(config.CLEAN_NSL_KDD, index=False)
    print(f"[clean_nsl_kdd] saved cleaned data to {config.CLEAN_NSL_KDD}")

    return X_clean, y


def get_feature_types(X_clean: pd.DataFrame) -> tuple:
    """Returns (categorical_cols, numeric_cols). categorical = protocol_type,
    service, flag. Used by member 2 to build the preprocessor"""

    categorical_cols = [c for c in ["protocol_type", "service", "flag"]
                        if c in X_clean.columns]
    numeric_cols = [c for c in X_clean.columns if c not in categorical_cols]

    print(f"[get_feature_types] {len(categorical_cols)} categorical: {categorical_cols}")
    print(f"[get_feature_types] {len(numeric_cols)} numeric columns")

    return categorical_cols, numeric_cols


def summarize_clean(X_clean: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Shape, dtypes, missing counts and class distribution BEFORE any split"""

    print("=" * 60)
    print("NSL-KDD Clean Summary")
    print("=" * 60)
    print(f"Features shape : {X_clean.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"Missing values : {X_clean.isnull().sum().sum()}")
    print()

    # Dtypes summary
    dtype_summary = X_clean.dtypes.value_counts()
    print("Column dtypes:")
    for dtype, count in dtype_summary.items():
        print(f"  {dtype}: {count} columns")
    print()

    # Class distribution
    print("Class distribution (target = 'label'):")
    class_dist = y.value_counts()
    for cls, cnt in class_dist.items():
        print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")
    print()

    # Build a summary DataFrame for the report
    summary = pd.DataFrame({
        "column": X_clean.columns,
        "dtype": X_clean.dtypes.values,
        "missing": X_clean.isnull().sum().values,
        "unique": X_clean.nunique().values,
    })

    return summary

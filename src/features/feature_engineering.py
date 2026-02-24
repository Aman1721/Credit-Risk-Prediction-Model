# src/features/feature_engineering.py
# ─────────────────────────────────────────────────────────────
# All feature transformations applied to raw loan data.
# Returns an enriched DataFrame ready for model training.
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import FEATURE_CONFIG


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Financial ratio features derived from raw columns."""
    df['loan_to_income_ratio']    = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_payment_est']     = df['loan_amount'] / df['loan_term']
    df['payment_to_income_ratio'] = (df['monthly_payment_est'] * 12) / (df['annual_income'] + 1)
    df['credit_util_proxy']       = df['dti_ratio'] * df['num_credit_lines']
    return df


def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary risk indicator flags."""
    df['has_delinquency']    = (df['delinq_2yrs'] > 0).astype(int)
    df['high_dti']           = (df['dti_ratio'] > 0.43).astype(int)
    df['multiple_inquiries'] = (df['num_inquiries'] >= 3).astype(int)
    df['short_employment']   = (df['employment_years'] < 2).astype(int)
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform skewed numeric columns."""
    for col in FEATURE_CONFIG['log_cols']:
        df[f'log_{col}'] = np.log1p(df[col])
    return df


def add_categorical_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bin numeric columns into meaningful categorical groups."""
    cfg = FEATURE_CONFIG

    df['credit_band'] = pd.cut(
        df['credit_score'],
        bins=cfg['credit_bins'],
        labels=cfg['credit_labels']
    ).astype(str)

    df['age_group'] = pd.cut(
        df['age'],
        bins=cfg['age_bins'],
        labels=cfg['age_labels']
    ).astype(str)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns."""
    cat_cols = FEATURE_CONFIG['cat_cols']
    # Only encode columns that actually exist in df
    cols_to_encode = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df


def run_feature_engineering(df: pd.DataFrame,
                              encode: bool = True) -> pd.DataFrame:
    """
    Master function: applies all feature engineering steps in order.

    Parameters
    ----------
    df     : Raw loan DataFrame
    encode : Whether to one-hot encode categoricals (True for training)

    Returns
    -------
    Enriched DataFrame with engineered features
    """
    print("[FeatureEngineering] Running transformations...")
    df = df.copy()
    df = add_ratio_features(df)
    df = add_risk_flags(df)
    df = add_log_transforms(df)
    df = add_categorical_bins(df)

    if encode:
        df = encode_categoricals(df)

    print(f"[FeatureEngineering] Features after engineering: {df.shape[1]} columns")
    return df


def get_feature_columns(df_encoded: pd.DataFrame) -> list:
    """Return feature column names (excludes target)."""
    target = FEATURE_CONFIG['target_col']
    return [c for c in df_encoded.columns if c != target]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data.data_generator import load_or_generate

    df_raw = load_or_generate()
    df_eng = run_feature_engineering(df_raw)
    print(df_eng.head(3))
    print(f"\nNew features created:\n{[c for c in df_eng.columns if c not in df_raw.columns]}")

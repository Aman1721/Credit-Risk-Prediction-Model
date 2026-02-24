# src/data/data_generator.py
# ─────────────────────────────────────────────────────────────
# Generates a realistic synthetic credit/loan dataset and
# saves it to data/raw_loans.csv
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import DATA_CONFIG


def generate_credit_data(n: int = DATA_CONFIG["n_samples"],
                          seed: int = DATA_CONFIG["random_seed"]) -> pd.DataFrame:
    """
    Generate synthetic loan application data with realistic
    default probability derived from financial attributes.

    Parameters
    ----------
    n    : Number of loan records to generate
    seed : Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with raw loan features + binary 'default' label
    """
    np.random.seed(seed)

    # ── Raw Features ─────────────────────────────────────────
    age              = np.random.randint(21, 70, n)
    income           = np.random.lognormal(10.8, 0.6, n).astype(int)
    loan_amount      = np.random.lognormal(9.5, 0.8, n).astype(int)
    loan_term        = np.random.choice([12, 24, 36, 48, 60], n)
    credit_score     = np.clip(np.random.normal(650, 80, n), 300, 850).astype(int)
    num_credit_lines = np.random.randint(1, 15, n)
    delinq_2yrs      = np.random.choice([0, 1, 2, 3, 4], n, p=[0.65, 0.18, 0.10, 0.05, 0.02])
    dti_ratio        = np.clip(np.random.normal(0.35, 0.15, n), 0.05, 0.95)
    employment_years = np.clip(np.random.normal(7, 5, n), 0, 40)
    home_ownership   = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n, p=[0.45, 0.20, 0.35])
    loan_purpose     = np.random.choice(
        ['debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'other'],
        n, p=[0.45, 0.20, 0.15, 0.10, 0.10]
    )
    num_inquiries = np.random.poisson(1.2, n)

    # ── Default Label Engineering ─────────────────────────────
    # Logistic model: higher score = lower risk
    logit = (
        -4.5
        + 0.02  * (700 - credit_score) / 100
        + 1.80  * dti_ratio
        + 0.50  * (delinq_2yrs > 0).astype(float)
        + 0.30  * (delinq_2yrs > 1).astype(float)
        - 0.30  * np.log(income / 50000 + 0.1)
        + 0.40  * np.log(loan_amount / 10000 + 0.1)
        - 0.02  * employment_years
        + 0.10  * (home_ownership == 'RENT').astype(float)
        + 0.15  * num_inquiries
        + np.random.normal(0, 0.5, n)     # noise
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = (np.random.rand(n) < prob_default).astype(int)

    df = pd.DataFrame({
        'age'             : age,
        'annual_income'   : income,
        'loan_amount'     : loan_amount,
        'loan_term'       : loan_term,
        'credit_score'    : credit_score,
        'num_credit_lines': num_credit_lines,
        'delinq_2yrs'     : delinq_2yrs,
        'dti_ratio'       : dti_ratio,
        'employment_years': employment_years,
        'home_ownership'  : home_ownership,
        'loan_purpose'    : loan_purpose,
        'num_inquiries'   : num_inquiries,
        'default'         : default,
    })

    return df


def load_or_generate(save: bool = True) -> pd.DataFrame:
    """
    Load dataset from CSV if it exists, otherwise generate and save.
    """
    path = DATA_CONFIG["raw_file"]
    if os.path.exists(path):
        print(f"[DataGenerator] Loading existing dataset from {path}")
        df = pd.read_csv(path)
    else:
        print(f"[DataGenerator] Generating {DATA_CONFIG['n_samples']} loan records...")
        df = generate_credit_data()
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"[DataGenerator] Saved to {path}")

    print(f"[DataGenerator] Shape: {df.shape} | Default rate: {df['default'].mean():.2%}")
    return df


if __name__ == "__main__":
    df = load_or_generate()
    print(df.head())
    print(df.describe().round(2))

# tests/test_pipeline.py
# ─────────────────────────────────────────────────────────────
# Unit tests for each module in the credit risk pipeline.
# Run with: python tests/test_pipeline.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_generator       import generate_credit_data
from src.features.feature_engineering import (
    run_feature_engineering, get_feature_columns,
    add_ratio_features, add_risk_flags, add_log_transforms, add_categorical_bins
)
from src.models.train_models       import build_models, cross_validate_model
from src.evaluation.evaluator      import threshold_simulation, get_optimal_thresholds
from src.utils.helpers             import (
    split_features_target, train_test_split_stratified, scale_features, align_columns
)


PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
results = []


def test(name, fn):
    try:
        fn()
        print(f"{PASS} — {name}")
        results.append(True)
    except Exception as e:
        print(f"{FAIL} — {name}  [{e}]")
        results.append(False)


# ── Data Generator Tests ─────────────────────────────────────
def test_data_shape():
    df = generate_credit_data(n=200)
    assert df.shape == (200, 13), f"Expected (200,13), got {df.shape}"

def test_data_target_binary():
    df = generate_credit_data(n=500)
    assert set(df['default'].unique()).issubset({0, 1}), "Target not binary"

def test_data_no_nulls():
    df = generate_credit_data(n=300)
    assert df.isnull().sum().sum() == 0, "Null values found in raw data"

def test_credit_score_range():
    df = generate_credit_data(n=500)
    assert df['credit_score'].between(300, 850).all(), "Credit scores out of range"


# ── Feature Engineering Tests ────────────────────────────────
def test_ratio_features():
    df = generate_credit_data(n=100)
    df = add_ratio_features(df)
    assert 'loan_to_income_ratio' in df.columns
    assert 'monthly_payment_est' in df.columns
    assert 'payment_to_income_ratio' in df.columns

def test_risk_flags():
    df = generate_credit_data(n=100)
    df = add_risk_flags(df)
    assert 'has_delinquency' in df.columns
    assert df['has_delinquency'].isin([0, 1]).all()

def test_log_transforms():
    df = generate_credit_data(n=100)
    df = add_log_transforms(df)
    assert 'log_annual_income' in df.columns
    assert 'log_loan_amount' in df.columns
    assert (df['log_annual_income'] >= 0).all()

def test_categorical_bins():
    df = generate_credit_data(n=200)
    df = add_categorical_bins(df)
    assert 'credit_band' in df.columns
    assert 'age_group' in df.columns

def test_full_feature_engineering():
    df = generate_credit_data(n=100)
    df_eng = run_feature_engineering(df, encode=True)
    assert df_eng.shape[1] > df.shape[1], "Engineering should add columns"
    assert 'default' in df_eng.columns

def test_feature_columns():
    df = generate_credit_data(n=100)
    df_eng = run_feature_engineering(df, encode=True)
    cols = get_feature_columns(df_eng)
    assert 'default' not in cols


# ── Preprocessing Tests ──────────────────────────────────────
def test_split_features_target():
    df = generate_credit_data(n=200)
    df = run_feature_engineering(df, encode=True)
    X, y, cols = split_features_target(df)
    assert 'default' not in X.columns
    assert len(y) == 200
    assert len(cols) == X.shape[1]

def test_train_test_split():
    df  = generate_credit_data(n=500)
    df  = run_feature_engineering(df, encode=True)
    X, y, _ = split_features_target(df)
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y)
    assert len(X_tr) + len(X_te) == 500
    assert abs(y_tr.mean() - y_te.mean()) < 0.05  # similar default rates

def test_scale_features():
    df  = generate_credit_data(n=300)
    df  = run_feature_engineering(df, encode=True)
    X, y, _ = split_features_target(df)
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y)
    X_tr_sc, X_te_sc, scaler = scale_features(X_tr, X_te)
    assert X_tr_sc.shape == X_tr.shape
    assert abs(X_tr_sc.mean()) < 0.1   # roughly zero-centered

def test_align_columns():
    feature_cols = ['a', 'b', 'c', 'd']
    df_new = pd.DataFrame([[1, 2]], columns=['a', 'b'])
    aligned = align_columns(df_new, feature_cols)
    assert list(aligned.columns) == feature_cols
    assert aligned['c'].iloc[0] == 0


# ── Model Tests ──────────────────────────────────────────────
def test_build_models():
    models = build_models()
    assert len(models) == 3
    for name, m in models.items():
        assert hasattr(m, 'fit') and hasattr(m, 'predict_proba')

def test_model_fits_and_predicts():
    df  = generate_credit_data(n=400)
    df  = run_feature_engineering(df, encode=True)
    X, y, _ = split_features_target(df)
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y)
    models = build_models()
    rf = models['Random Forest']
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:, 1]
    assert probs.shape == (len(X_te),)
    assert probs.min() >= 0 and probs.max() <= 1


# ── Evaluation Tests ─────────────────────────────────────────
def test_threshold_simulation():
    y_test = pd.Series(np.random.randint(0, 2, 200))
    y_prob = np.random.rand(200)
    df = threshold_simulation(y_test, y_prob)
    assert 'threshold' in df.columns
    assert 'f1' in df.columns
    assert 'simulated_profit' in df.columns
    assert len(df) > 0

def test_optimal_thresholds():
    y_test = pd.Series(np.random.randint(0, 2, 200))
    y_prob = np.random.rand(200)
    df = threshold_simulation(y_test, y_prob)
    t_f1, t_profit = get_optimal_thresholds(df)
    assert 0 < t_f1 < 1
    assert 0 < t_profit < 1


# ── Run all tests ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  CREDIT RISK PROJECT — UNIT TESTS")
    print("=" * 55)

    print("\n[Data Generator]")
    test("Shape (200 rows, 13 cols)",    test_data_shape)
    test("Target is binary {0,1}",       test_data_target_binary)
    test("No null values",               test_data_no_nulls)
    test("Credit score in [300, 850]",   test_credit_score_range)

    print("\n[Feature Engineering]")
    test("Ratio features added",         test_ratio_features)
    test("Risk flags are binary",        test_risk_flags)
    test("Log transforms non-negative",  test_log_transforms)
    test("Categorical bins created",     test_categorical_bins)
    test("Full pipeline adds columns",   test_full_feature_engineering)
    test("Target excluded from features",test_feature_columns)

    print("\n[Preprocessing / Utils]")
    test("split_features_target works",  test_split_features_target)
    test("Stratified split preserves rate", test_train_test_split)
    test("StandardScaler zero-centers",  test_scale_features)
    test("align_columns fills missing",  test_align_columns)

    print("\n[Models]")
    test("3 models built",               test_build_models)
    test("RF fits and outputs probs",    test_model_fits_and_predicts)

    print("\n[Evaluation]")
    test("Threshold df has all cols",    test_threshold_simulation)
    test("Optimal thresholds in (0,1)",  test_optimal_thresholds)

    total  = len(results)
    passed = sum(results)
    print(f"\n{'='*55}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'='*55}\n")

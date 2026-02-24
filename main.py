import os
import sys
import numpy as np

# ── Ensure project root is on path ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_generator       import load_or_generate
from src.features.feature_engineering import (
    run_feature_engineering, get_feature_columns
)
from src.models.train_models       import (
    build_models, train_and_evaluate, get_best_model
)
from src.evaluation.evaluator      import (
    run_full_evaluation, score_applicant, get_optimal_thresholds
)
from src.evaluation.visualizer     import build_dashboard
from src.utils.helpers             import (
    split_features_target, train_test_split_stratified,
    scale_features, ensure_output_dirs, print_section
)


def main():
    ensure_output_dirs()

    # ────────────────────────────────────────────────────────
    # STEP 1 — Load / Generate Data
    # ────────────────────────────────────────────────────────
    print_section("STEP 1 — DATA LOADING")
    df_raw = load_or_generate(save=True)

    # ────────────────────────────────────────────────────────
    # STEP 2 — Feature Engineering
    # ────────────────────────────────────────────────────────
    print_section("STEP 2 — FEATURE ENGINEERING")
    df_eng = run_feature_engineering(df_raw, encode=True)

    # ────────────────────────────────────────────────────────
    # STEP 3 — Split & Scale
    # ────────────────────────────────────────────────────────
    print_section("STEP 3 — TRAIN / TEST SPLIT")
    X, y, feature_cols = split_features_target(df_eng)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)

    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")
    print(f"  Features : {len(feature_cols)}")

    # ────────────────────────────────────────────────────────
    # STEP 4 — Model Training
    # ────────────────────────────────────────────────────────
    models  = build_models()
    results = train_and_evaluate(
        models,
        X_train, y_train,
        X_train_sc, y_train,
        X_test, y_test,
        X_test_sc
    )

    # ────────────────────────────────────────────────────────
    # STEP 5 — Best Model & Evaluation
    # ────────────────────────────────────────────────────────
    best_name, best_result = get_best_model(results)
    thresh_df, t_profit    = run_full_evaluation(best_name, best_result, y_test)
    t_f1, _                = get_optimal_thresholds(thresh_df)

    y_pred_final = (best_result['y_prob'] >= t_profit).astype(int)

    # ────────────────────────────────────────────────────────
    # STEP 6 — Visualization Dashboard
    # ────────────────────────────────────────────────────────
    print_section("STEP 6 — GENERATING DASHBOARD")
    rf_model  = results['Random Forest']['model']
    dashboard = build_dashboard(
        df_raw      = df_raw,
        y           = y,
        results     = results,
        y_test      = y_test,
        thresh_df   = thresh_df,
        t_f1        = t_f1,
        t_profit    = t_profit,
        y_pred_final= y_pred_final,
        feature_cols= feature_cols,
        rf_model    = rf_model,
    )

    # ────────────────────────────────────────────────────────
    # STEP 7 — Live Scoring Demo
    # ────────────────────────────────────────────────────────
    print_section("STEP 7 — LOAN APPLICATION SCORING")
    best_model = best_result['model']

    applicants = [
        {
            "label": "Prime Borrower",
            "data" : {
                'age': 35, 'annual_income': 85000, 'loan_amount': 15000,
                'loan_term': 36, 'credit_score': 720, 'num_credit_lines': 5,
                'delinq_2yrs': 0, 'dti_ratio': 0.25, 'employment_years': 8,
                'home_ownership': 'OWN', 'loan_purpose': 'home_improvement',
                'num_inquiries': 1,
            }
        },
        {
            "label": "High-Risk Borrower",
            "data" : {
                'age': 28, 'annual_income': 32000, 'loan_amount': 18000,
                'loan_term': 60, 'credit_score': 560, 'num_credit_lines': 3,
                'delinq_2yrs': 2, 'dti_ratio': 0.62, 'employment_years': 1,
                'home_ownership': 'RENT', 'loan_purpose': 'debt_consolidation',
                'num_inquiries': 5,
            }
        },
        {
            "label": "Mid-Tier Borrower",
            "data" : {
                'age': 45, 'annual_income': 55000, 'loan_amount': 10000,
                'loan_term': 36, 'credit_score': 650, 'num_credit_lines': 6,
                'delinq_2yrs': 1, 'dti_ratio': 0.38, 'employment_years': 5,
                'home_ownership': 'MORTGAGE', 'loan_purpose': 'major_purchase',
                'num_inquiries': 2,
            }
        },
    ]

    for applicant in applicants:
        result = score_applicant(
            raw_record   = applicant['data'],
            model        = best_model,
            feature_cols = feature_cols,
            threshold    = t_profit,
        )
        d = applicant['data']
        print(f"\n  ── {applicant['label']} ──")
        print(f"     Credit Score : {d['credit_score']}  |  Income: ${d['annual_income']:,}  |  DTI: {d['dti_ratio']:.0%}")
        print(f"     Default Prob : {result['default_probability']:.2%}")
        print(f"     Risk Band    : {result['risk_band']}")
        print(f"     Decision     : ★  {result['decision']}  ★")

    print_section("PIPELINE COMPLETE ✅")
    print(f"  Dashboard  → {dashboard}")
    print(f"  Report     → outputs/reports/threshold_simulation.csv")
    print(f"  Dataset    → data/raw_loans.csv\n")


if __name__ == "__main__":
    main()

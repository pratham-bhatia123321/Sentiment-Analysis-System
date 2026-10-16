"""
Hyperparameter Optimization Module.

- GridSearchCV for ML models (Logistic Regression, XGBoost, Random Forest)
- Deep learning models already use EarlyStopping + ModelCheckpoint during training
"""

import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils import print_header, print_info, print_success, print_subheader, print_metric


def tune_logistic_regression(X_train, y_train, param_grid, save_path):
    """Tune Logistic Regression via GridSearchCV (3-fold CV)."""
    print_subheader("TUNING: LOGISTIC REGRESSION")
    print_info(f"Parameter grid: {param_grid}")

    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
    )

    print_info("Running GridSearchCV (3-fold)...")
    grid.fit(X_train, y_train)

    print_metric("Best Parameters", grid.best_params_)
    print_metric("Best CV F1 Score", f"{grid.best_score_:.4f}")
    print_metric("Total Fits", f"{grid.cv_results_['mean_test_score'].shape[0] * 3}")

    joblib.dump(grid.best_estimator_, save_path)
    print_success(f"Tuned model saved: {save_path}")

    return grid.best_estimator_, grid.best_score_


def tune_xgboost(X_train, y_train, param_grid, save_path):
    """Tune XGBoost via GridSearchCV (3-fold CV)."""
    print_subheader("TUNING: XGBOOST")
    print_info(f"Parameter grid: {param_grid}")

    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
    )

    print_info("Running GridSearchCV (3-fold)...")
    grid.fit(X_train, y_train)

    print_metric("Best Parameters", grid.best_params_)
    print_metric("Best CV F1 Score", f"{grid.best_score_:.4f}")

    joblib.dump(grid.best_estimator_, save_path)
    print_success(f"Tuned model saved: {save_path}")

    return grid.best_estimator_, grid.best_score_


def tune_random_forest(X_train, y_train, param_grid, save_path):
    """Tune Random Forest via GridSearchCV (3-fold CV)."""
    print_subheader("TUNING: RANDOM FOREST")
    print_info(f"Parameter grid: {param_grid}")

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
    )

    print_info("Running GridSearchCV (3-fold)...")
    grid.fit(X_train, y_train)

    print_metric("Best Parameters", grid.best_params_)
    print_metric("Best CV F1 Score", f"{grid.best_score_:.4f}")

    joblib.dump(grid.best_estimator_, save_path)
    print_success(f"Tuned model saved: {save_path}")

    return grid.best_estimator_, grid.best_score_


def run_hyperparameter_tuning(tfidf_train, y_train, w2v_train, config):
    """Run hyperparameter tuning for all ML models.

    Args:
        tfidf_train: TF-IDF training features
        y_train: Training labels
        w2v_train: Word2Vec training features
        config: dict with 'lr_params', 'xgb_params', 'rf_params',
                'lr_path', 'xgb_path', 'rf_path'

    Returns:
        dict mapping model_name -> {'model': ..., 'cv_score': ..., 'type': ...}
    """
    print_header("STEP 7: HYPERPARAMETER OPTIMIZATION")

    results = {}

    # 1) Logistic Regression
    lr_model, lr_score = tune_logistic_regression(
        tfidf_train, y_train,
        config['lr_params'], config['lr_path'],
    )
    results['Logistic Regression'] = {
        'model': lr_model, 'cv_score': lr_score, 'type': 'tfidf',
    }

    # 2) XGBoost
    xgb_model, xgb_score = tune_xgboost(
        tfidf_train, y_train,
        config['xgb_params'], config['xgb_path'],
    )
    results['XGBoost'] = {
        'model': xgb_model, 'cv_score': xgb_score, 'type': 'tfidf',
    }

    # 3) Random Forest
    rf_model, rf_score = tune_random_forest(
        w2v_train, y_train,
        config['rf_params'], config['rf_path'],
    )
    results['Random Forest'] = {
        'model': rf_model, 'cv_score': rf_score, 'type': 'w2v',
    }

    # Summary
    print_subheader("HYPERPARAMETER TUNING SUMMARY")
    for name, data in results.items():
        print_metric(f"{name} Best CV F1", f"{data['cv_score']:.4f}")

    print_info("\nDeep Learning models used EarlyStopping + ModelCheckpoint")
    print_info("(hyperparameters optimized via early stopping on validation loss)")

    return results

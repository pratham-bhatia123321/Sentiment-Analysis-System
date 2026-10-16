"""
Advanced Sentiment Analysis System
====================================
Main execution pipeline that orchestrates the complete ML/DL workflow.

Stages:
    1. Dataset Loading (IMDB 50K reviews)
    2. Text Preprocessing (NLP pipeline)
    3. Exploratory Data Analysis (7 visualizations)
    4. Feature Engineering (TF-IDF, Word2Vec, DL sequences)
    5. Model Training (5 models: LR, XGBoost, RF, LSTM, BiLSTM)
    6. Model Evaluation (metrics, confusion matrices, comparison)
    7. Hyperparameter Optimization (GridSearchCV + EarlyStopping)
    8. Model Saving
    9. Prediction Pipeline Testing

Usage:
    python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import *
from src.utils import print_header, print_info, print_success, print_banner
from src.data_loader import load_imdb_dataset, split_dataset
from src.preprocessing import preprocess_dataset
from src.eda import run_eda
from src.feature_engineering import (
    create_tfidf_features,
    create_word2vec_features,
    prepare_dl_sequences,
)
from src.models import (
    train_logistic_regression,
    train_xgboost,
    train_random_forest,
    train_lstm,
    train_bilstm,
)
from src.evaluation import evaluate_all_models, plot_training_history
from src.tuning import run_hyperparameter_tuning
from src.prediction import SentimentPredictor, test_predictions


def main():
    """Execute the complete sentiment analysis pipeline end-to-end."""

    start_time = time.time()

    # ─── Banner ──────────────────────────────────────────
    print_banner()

    # ─── Create Directories ──────────────────────────────
    for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
                      os.path.join(OUTPUTS_DIR, 'plots')]:
        os.makedirs(directory, exist_ok=True)

    # ═════════════════════════════════════════════════════════
    # STEP 1 : LOAD DATASET
    # ═════════════════════════════════════════════════════════
    df = load_imdb_dataset(DATA_DIR, IMDB_URL, DATASET_CSV)

    # ═════════════════════════════════════════════════════════
    # STEP 2 : TEXT PREPROCESSING
    # ═════════════════════════════════════════════════════════
    df = preprocess_dataset(df)

    # ═════════════════════════════════════════════════════════
    # STEP 3 : EXPLORATORY DATA ANALYSIS
    # ═════════════════════════════════════════════════════════
    run_eda(df, OUTPUTS_DIR)

    # ═════════════════════════════════════════════════════════
    # STEP 4 : FEATURE ENGINEERING
    # ═════════════════════════════════════════════════════════
    print_header("STEP 4: FEATURE ENGINEERING")

    X_train, X_test, y_train, y_test = split_dataset(df, TEST_SIZE, RANDOM_STATE)

    # A) TF-IDF
    tfidf_train, tfidf_test, vectorizer = create_tfidf_features(
        X_train, X_test, TFIDF_MAX_FEATURES, TFIDF_PATH,
    )

    # B) Word2Vec
    w2v_train, w2v_test, w2v_model = create_word2vec_features(
        X_train, X_test,
        W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_WORKERS,
        W2V_MODEL_PATH,
    )

    # C) DL Sequences
    dl_train, dl_test, dl_tokenizer = prepare_dl_sequences(
        X_train, X_test,
        MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH,
        TOKENIZER_PATH,
    )

    # Convert labels to numpy arrays
    y_train_np = y_train.values
    y_test_np = y_test.values

    # ═════════════════════════════════════════════════════════
    # STEP 5 : MODEL TRAINING
    # ═════════════════════════════════════════════════════════
    print_header("STEP 5: MODEL TRAINING")

    # Model 1: Logistic Regression
    lr_model = train_logistic_regression(tfidf_train, y_train_np, LR_MODEL_PATH)

    # Model 2: XGBoost
    xgb_model = train_xgboost(tfidf_train, y_train_np, XGB_MODEL_PATH)

    # Model 3: Random Forest
    rf_model = train_random_forest(w2v_train, y_train_np, RF_MODEL_PATH)

    # Model 4: LSTM
    vocab_size = min(MAX_NUM_WORDS, len(dl_tokenizer.word_index) + 1)
    lstm_model, lstm_history = train_lstm(
        dl_train, y_train_np, LSTM_MODEL_PATH,
        vocab_size, MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
    )
    plot_training_history(lstm_history, 'LSTM',
                          os.path.join(OUTPUTS_DIR, 'plots'))

    # Model 5: Bidirectional LSTM
    bilstm_model, bilstm_history = train_bilstm(
        dl_train, y_train_np, BILSTM_MODEL_PATH,
        vocab_size, MAX_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
    )
    plot_training_history(bilstm_history, 'BiLSTM',
                          os.path.join(OUTPUTS_DIR, 'plots'))

    # ═════════════════════════════════════════════════════════
    # STEP 6 : MODEL EVALUATION
    # ═════════════════════════════════════════════════════════
    # Generate predictions on test set
    print_info("Generating predictions on test set...")
    lr_pred = lr_model.predict(tfidf_test)
    xgb_pred = xgb_model.predict(tfidf_test)
    rf_pred = rf_model.predict(w2v_test)
    lstm_pred = (lstm_model.predict(dl_test, verbose=0) > 0.5).astype(int).flatten()
    bilstm_pred = (bilstm_model.predict(dl_test, verbose=0) > 0.5).astype(int).flatten()

    models_eval = {
        'TF-IDF + LR':          (y_test_np, lr_pred),
        'TF-IDF + XGBoost':     (y_test_np, xgb_pred),
        'W2V + Random Forest':  (y_test_np, rf_pred),
        'LSTM':                 (y_test_np, lstm_pred),
        'BiLSTM':               (y_test_np, bilstm_pred),
    }

    results_df = evaluate_all_models(models_eval, OUTPUTS_DIR)

    # ═════════════════════════════════════════════════════════
    # STEP 7 : HYPERPARAMETER OPTIMIZATION
    # ═════════════════════════════════════════════════════════
    tuning_config = {
        'lr_params': LR_PARAM_GRID,
        'xgb_params': XGB_PARAM_GRID,
        'rf_params': RF_PARAM_GRID,
        'lr_path': LR_MODEL_PATH,
        'xgb_path': XGB_MODEL_PATH,
        'rf_path': RF_MODEL_PATH,
    }

    tuned = run_hyperparameter_tuning(
        tfidf_train, y_train_np, w2v_train, tuning_config,
    )

    # Re-evaluate tuned models vs DL models
    print_header("STEP 7b: TUNED MODEL EVALUATION")
    print_info("Evaluating tuned ML models alongside DL models...")

    tuned_lr_pred  = tuned['Logistic Regression']['model'].predict(tfidf_test)
    tuned_xgb_pred = tuned['XGBoost']['model'].predict(tfidf_test)
    tuned_rf_pred  = tuned['Random Forest']['model'].predict(w2v_test)

    tuned_eval = {
        'Tuned LR':      (y_test_np, tuned_lr_pred),
        'Tuned XGBoost': (y_test_np, tuned_xgb_pred),
        'Tuned RF':      (y_test_np, tuned_rf_pred),
        'LSTM':          (y_test_np, lstm_pred),
        'BiLSTM':        (y_test_np, bilstm_pred),
    }

    final_results = evaluate_all_models(tuned_eval, OUTPUTS_DIR)

    # ═════════════════════════════════════════════════════════
    # STEP 8 : SAVED MODELS SUMMARY
    # ═════════════════════════════════════════════════════════
    print_header("STEP 8: SAVED MODELS")
    print_info("All models and artifacts in 'models/' directory:\n")

    for fname in sorted(os.listdir(MODELS_DIR)):
        fpath = os.path.join(MODELS_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        if size_kb > 1024:
            size_str = f"{size_kb / 1024:.1f} MB"
        else:
            size_str = f"{size_kb:.1f} KB"
        print(f"    {fname:45s}  {size_str:>10s}")

    # ═════════════════════════════════════════════════════════
    # STEP 9 : PREDICTION PIPELINE TESTING
    # ═════════════════════════════════════════════════════════
    predictor_config = {
        'lr_path': LR_MODEL_PATH,
        'xgb_path': XGB_MODEL_PATH,
        'rf_path': RF_MODEL_PATH,
        'lstm_path': LSTM_MODEL_PATH,
        'bilstm_path': BILSTM_MODEL_PATH,
        'tfidf_path': TFIDF_PATH,
        'tokenizer_path': TOKENIZER_PATH,
        'w2v_path': W2V_MODEL_PATH,
        'w2v_size': W2V_VECTOR_SIZE,
        'max_seq_len': MAX_SEQUENCE_LENGTH,
    }

    predictor = SentimentPredictor(predictor_config)
    predictor.load_models()
    test_predictions(predictor)

    # ═════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    best_model = final_results.iloc[0]

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total Execution Time : {elapsed / 60:.1f} minutes")
    print(f"  Models Trained       : 5 (3 ML + 2 DL)")
    print(f"  Best Model           : {best_model['Model']} "
          f"(F1: {best_model['F1 Score']:.4f})")
    print(f"  Plots Saved          : outputs/plots/")
    print(f"  Models Saved         : models/")
    print(f"  Results CSV          : outputs/model_results.csv")
    print("=" * 70)
    print("  Run the Streamlit app:  streamlit run app.py")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

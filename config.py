"""
Configuration file for Advanced Sentiment Analysis System.
Centralizes all hyperparameters, file paths, and settings.
"""

import os

# ═══════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════
IMDB_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
DATASET_CSV = os.path.join(DATA_DIR, 'imdb_reviews.csv')

# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
TFIDF_MAX_FEATURES = 10000
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 4
MAX_SEQUENCE_LENGTH = 200
MAX_NUM_WORDS = 10000

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_SPLIT = 0.1

# ═══════════════════════════════════════════════════════════════
# MODEL SAVE PATHS
# ═══════════════════════════════════════════════════════════════
LR_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost.pkl')
RF_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest.pkl')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.keras')
BILSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'bilstm_model.keras')
TFIDF_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
W2V_MODEL_PATH = os.path.join(MODELS_DIR, 'word2vec_model.model')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'dl_tokenizer.pkl')
RESULTS_PATH = os.path.join(OUTPUTS_DIR, 'model_results.csv')

# ═══════════════════════════════════════════════════════════════
# HYPERPARAMETER GRIDS (for GridSearchCV)
# ═══════════════════════════════════════════════════════════════
LR_PARAM_GRID = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [1000],
}

XGB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8],
}

RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
}

"""
Model Training Module.

Trains 5 sentiment classification models:
    1. TF-IDF  + Logistic Regression
    2. TF-IDF  + XGBoost
    3. Word2Vec + Random Forest
    4. LSTM Neural Network
    5. Bidirectional LSTM Neural Network
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import print_info, print_success, print_subheader, print_metric


# ═══════════════════════════════════════════════════════════════
# MACHINE LEARNING MODELS
# ═══════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, save_path):
    """Train Logistic Regression on TF-IDF features."""
    print_subheader("MODEL 1: TF-IDF + LOGISTIC REGRESSION")

    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )

    print_info("Training Logistic Regression...")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print_metric("Training Accuracy", f"{train_acc:.4f}")
    print_metric("Coefficients Shape", model.coef_.shape)

    joblib.dump(model, save_path)
    print_success(f"Model saved: {save_path}")

    return model


def train_xgboost(X_train, y_train, save_path):
    """Train XGBoost Classifier on TF-IDF features."""
    print_subheader("MODEL 2: TF-IDF + XGBOOST")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=0,
    )

    print_info("Training XGBoost (200 trees, depth=6)...")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print_metric("Training Accuracy", f"{train_acc:.4f}")
    print_metric("N Estimators", model.n_estimators)
    print_metric("Max Depth", model.max_depth)

    joblib.dump(model, save_path)
    print_success(f"Model saved: {save_path}")

    return model


def train_random_forest(X_train, y_train, save_path):
    """Train Random Forest on Word2Vec document vectors."""
    print_subheader("MODEL 3: WORD2VEC + RANDOM FOREST")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )

    print_info("Training Random Forest (200 trees)...")
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print_metric("Training Accuracy", f"{train_acc:.4f}")
    print_metric("N Estimators", model.n_estimators)

    joblib.dump(model, save_path)
    print_success(f"Model saved: {save_path}")

    return model


# ═══════════════════════════════════════════════════════════════
# DEEP LEARNING MODELS
# ═══════════════════════════════════════════════════════════════

def build_lstm_model(vocab_size, embedding_dim, max_length):
    """Build LSTM architecture.

    Architecture:
        Embedding(vocab_size, 128)
        SpatialDropout1D(0.2)
        LSTM(128)
        Dropout(0.3)
        Dense(64, relu)
        Dropout(0.2)
        Dense(1, sigmoid)
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        SpatialDropout1D(0.2),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_bilstm_model(vocab_size, embedding_dim, max_length):
    """Build Bidirectional LSTM architecture.

    Architecture:
        Embedding(vocab_size, 128)
        SpatialDropout1D(0.2)
        Bidirectional(LSTM(128))
        Dropout(0.3)
        Dense(64, relu)
        Dropout(0.2)
        Dense(1, sigmoid)
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_lstm(X_train, y_train, save_path, vocab_size, max_length,
               embedding_dim=128, batch_size=64, epochs=5):
    """Train LSTM model with EarlyStopping and ModelCheckpoint."""
    print_subheader("MODEL 4: LSTM NEURAL NETWORK")

    model = build_lstm_model(vocab_size, embedding_dim, max_length)

    print_info("LSTM Architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=2,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            save_path, monitor='val_accuracy',
            save_best_only=True, verbose=0,
        ),
    ]

    print_info("\nTraining LSTM...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print_success(f"LSTM model saved: {save_path}")
    return model, history


def train_bilstm(X_train, y_train, save_path, vocab_size, max_length,
                 embedding_dim=128, batch_size=64, epochs=5):
    """Train Bidirectional LSTM with EarlyStopping and ModelCheckpoint."""
    print_subheader("MODEL 5: BIDIRECTIONAL LSTM")

    model = build_bilstm_model(vocab_size, embedding_dim, max_length)

    print_info("BiLSTM Architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=2,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            save_path, monitor='val_accuracy',
            save_best_only=True, verbose=0,
        ),
    ]

    print_info("\nTraining Bidirectional LSTM...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print_success(f"BiLSTM model saved: {save_path}")
    return model, history

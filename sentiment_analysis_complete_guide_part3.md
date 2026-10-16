# 🎬 Advanced Sentiment Analysis System — Complete Interview Mastery Guide

# PART 3: Pipeline, Codebase, Hyperparameters & Mathematics (Sections 11–15)

---

## SECTION 11: PREDICTION PIPELINE

The prediction code lives in [prediction.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/prediction.py). It uses the `SentimentPredictor` class — a unified interface that loads all 5 models and runs any text through all of them.

### 11.1 SentimentPredictor Class Architecture

```python
class SentimentPredictor:
    def __init__(self, config):      # Store model paths
    def load_models(self):           # Load all 5 models + 3 feature extractors from disk
    def predict(self, text):         # Run text through all models
    def _predict_tfidf(self, ...)    # TF-IDF path (LR, XGBoost)
    def _predict_w2v(self, ...)      # Word2Vec path (Random Forest)
    def _predict_dl(self, ...)       # Deep Learning path (LSTM, BiLSTM)
```

### 11.2 Complete Trace: "I absolutely loved this movie."

Let's trace this review step by step.

#### Step 1: Raw input

```
"I absolutely loved this movie."
```

#### Step 2: Preprocessing — `preprocess_text()` called at [prediction.py line 79](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/prediction.py#L79)

```
lowercase:       "i absolutely loved this movie."
remove HTML:     "i absolutely loved this movie."  (no change)
remove URLs:     "i absolutely loved this movie."  (no change)
remove emojis:   "i absolutely loved this movie."  (no change)
remove punct:    "i absolutely loved this movie"   (period removed)
remove numbers:  "i absolutely loved this movie"   (no change)
remove special:  "i absolutely loved this movie"   (no change)
tokenize:        ["i", "absolutely", "loved", "this", "movie"]
stopwords:       ["absolutely", "loved", "movie"]
                 ("i" and "this" are stopwords and removed)
lemmatize:       ["absolutely", "loved", "movie"]
                 ("loved" stays "loved" — WordNetLemmatizer doesn't always
                  catch past tense verbs without POS tagging)
remove short:    ["absolutely", "loved", "movie"]  (all >1 char)
rejoin:          "absolutely loved movie"
```

**cleaned text = `"absolutely loved movie"`**

#### Step 3: Empty check

```python
if not cleaned.strip():
    return {name: {'sentiment': 'Unknown', 'confidence': 0.0} ...}
```

"absolutely loved movie" is not empty → proceed.

#### Step 4a: TF-IDF Path — `_predict_tfidf()` at [line 116](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/prediction.py#L116)

```python
features = self.vectorizer.transform(["absolutely loved movie"])
# → sparse matrix of shape (1, 10000)
# "absolutely" gets TF-IDF weight at its column index
# "loved" gets TF-IDF weight at its column index
# "movie" gets TF-IDF weight at its column index
# All other 9,997 columns = 0

proba = model.predict_proba(features)[0]
# → [0.12, 0.88]  (example: 12% negative, 88% positive)

prediction = int(proba[1] >= 0.5)  # 0.88 >= 0.5 → 1 (Positive)
confidence = max(proba)             # max(0.12, 0.88) = 0.88
```

**Logistic Regression output**: Positive, 88% confidence
**XGBoost output**: Same TF-IDF features, different model → e.g., Positive, 82% confidence

#### Step 4b: Word2Vec Path — `_predict_w2v()` at [line 124](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/prediction.py#L124)

```python
tokens = "absolutely loved movie".split()
# → ["absolutely", "loved", "movie"]

vecs = [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
# Look up each word's 100-dim vector
# If a word is NOT in vocabulary → skip it

doc_vec = np.mean(vecs, axis=0)
# Average the 3 vectors → (100,) single document vector

features = doc_vec.reshape(1, -1)
# → (1, 100) matrix

proba = model.predict_proba(features)[0]
# Random Forest: each of 200 trees votes
# → [0.15, 0.85]  (example)

prediction = int(proba[1] >= 0.5)  # Positive
confidence = max(proba)             # 0.85
```

**Random Forest output**: Positive, 85% confidence

#### Step 4c: Deep Learning Path — `_predict_dl()` at [line 140](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/prediction.py#L140)

```python
seq = self.dl_tokenizer.texts_to_sequences(["absolutely loved movie"])
# → [[547, 203, 42]]  (example word indices)

padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
# → [[547, 203, 42, 0, 0, 0, ..., 0]]  (padded to 200)

proba = model.predict(padded, verbose=0)[0][0]
# LSTM/BiLSTM sigmoid output → 0.74  (example)

prediction = int(proba >= 0.5)  # 0.74 >= 0.5 → 1 (Positive)
confidence = float(proba)       # Since prediction=1, confidence = proba = 0.74
# If prediction were 0, confidence = 1 - proba
```

**LSTM output**: Positive, 74% confidence
**BiLSTM output**: Positive, 79% confidence (typically higher than LSTM)

#### Step 5: Final results dictionary

```python
{
    'Logistic Regression': {'sentiment': 'Positive', 'confidence': 0.88},
    'XGBoost':             {'sentiment': 'Positive', 'confidence': 0.82},
    'Random Forest':       {'sentiment': 'Positive', 'confidence': 0.85},
    'LSTM':                {'sentiment': 'Positive', 'confidence': 0.74},
    'BiLSTM':              {'sentiment': 'Positive', 'confidence': 0.79},
}
```

### 11.3 How OOV (Out-of-Vocabulary) Words Are Handled

Suppose the review contains a rare word like "supercalifragilistic":

| Feature Pipeline | OOV Handling | Implementation |
|---|---|---|
| **TF-IDF** | Word is not in vocabulary → gets **zero weight** (ignored) | `vectorizer.transform()` only uses known vocabulary columns |
| **Word2Vec** | Word is not in `w2v_model.wv` → **skipped** in averaging | `if w in self.w2v_model.wv` check at [line 127](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/prediction.py#L127) |
| **Keras Tokenizer** | Word is not in top 10K → mapped to index `1` (`<OOV>`) | `oov_token='<OOV>'` set at [feature_engineering.py line 131](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py#L131) |

**If ALL words are OOV** (Word2Vec path): Returns a zero vector `np.zeros(100)`, which will likely predict close to 50/50.

> **What you should remember for the interview:**
> - The prediction pipeline has 3 separate internal paths depending on the model type
> - ALL paths start with the SAME preprocessing step
> - OOV words are handled gracefully: ignored (TF-IDF), skipped (W2V), or mapped to `<OOV>` index (DL)
> - Confidence is calculated differently for ML models (`max(proba)`) vs DL models (sigmoid output directly)
> - The SentimentPredictor class provides a unified interface — clean separation of concerns

---

## SECTION 12: STREAMLIT APPLICATION

The web application is in [app.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/app.py).

### 12.1 Application Structure

```
┌─────────────────────────────────────────────────┐
│              HEADER (gradient banner)            │
├──────────────────┬──────────────────────────────┤
│   TEXT INPUT     │     MODEL INFO PANEL         │
│   (textarea)     │  (metrics: models loaded,    │
│                  │   selected model,             │
│   [Analyze]      │   input length)              │
├──────────────────┴──────────────────────────────┤
│           PRIMARY RESULT CARD                    │
│  (Positive/Negative with avg confidence)         │
├─────────────────────────────────────────────────┤
│         PER-MODEL RESULTS (cards in columns)     │
├─────────────────────────────────────────────────┤
│         CONFIDENCE BAR CHART (matplotlib)        │
├─────────────────────────────────────────────────┤
│         PREPROCESSING DETAILS (expandable)       │
├─────────────────────────────────────────────────┤
│         MODEL COMPARISON TABLE + CHART           │
├─────────────────────────────────────────────────┤
│   SIDEBAR:                                       │
│   - Model selector dropdown                      │
│   - Quick example buttons                        │
│   - About section                                │
└─────────────────────────────────────────────────┘
```

### 12.2 Model Loading — `@st.cache_resource`

```python
@st.cache_resource(show_spinner="Loading ML/DL models...")
def load_all_models():
```

At [app.py line 142](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/app.py#L142):

- `@st.cache_resource` is a Streamlit decorator that loads models **once** and caches them across all user sessions
- Without caching, models would reload on every page refresh (slow — especially the 119MB Random Forest and 18MB BiLSTM)
- The function loads:
  - 3 ML models via `joblib.load()` (Logistic Regression, XGBoost, Random Forest)
  - 2 DL models via `load_model()` (LSTM, BiLSTM)
  - 3 feature extractors (TF-IDF vectorizer, Keras tokenizer, Word2Vec model)
- Returns: `(models_dict, vectorizer, dl_tokenizer, w2v_model)`

### 12.3 User Input Flow

1. User types a review in the `st.text_area` widget
2. Or clicks a sidebar example button (sets `st.session_state['input_text']`)
3. Clicks "🔍 Analyze Sentiment" button
4. `preprocess_text(user_input)` is called — same function as training
5. If cleaned text is empty → warning message
6. Otherwise → runs prediction through selected model(s)

### 12.4 Prediction and Display

```python
# Determine models to run
if selected_model == 'All Models':
    models_to_run = models  # all 5
else:
    models_to_run = {selected_model: models[selected_model]}  # just 1

# Run predictions
for name, model in models_to_run.items():
    pred, conf = predict_with_model(model, name, cleaned, ...)
```

The `predict_with_model()` function at [app.py line 172](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/app.py#L172) duplicates the logic from `SentimentPredictor` (it's a standalone function, not using the class). This is a design choice for the Streamlit app — the cached models are passed directly instead of through a class instance.

### 12.5 Majority Voting

When "All Models" is selected:
```python
sentiments = [r['sentiment'] for r in results.values()]
majority = max(set(sentiments), key=sentiments.count)
avg_confidence = np.mean([r['confidence'] for r in results.values()])
```

If 3 models say Positive and 2 say Negative → Overall: **Positive**

### 12.6 Visualization Components

| Component | Library | Purpose |
|---|---|---|
| Gradient header | HTML/CSS (`unsafe_allow_html=True`) | Premium visual appearance |
| Result cards | HTML/CSS (green gradient = positive, red = negative) | Clear visual feedback |
| Model cards | HTML/CSS | Per-model results display |
| Confidence bar chart | matplotlib (`st.pyplot(fig)`) | Compare model confidences |
| Model comparison table | pandas DataFrame (`st.dataframe()`) | Historical benchmark data |
| Comparison chart image | Static PNG from `outputs/plots/` (`st.image()`) | Pre-generated during training |
| Preprocessing expander | `st.expander()` | Shows before/after text |

### 12.7 Features ACTUALLY Implemented vs README Claims

| Feature | README Claims | Actually Implemented |
|---|---|---|
| Real-time single prediction | ✅ Yes | ✅ Yes — fully working |
| Batch CSV processing | ✅ Mentioned | ❌ **Not implemented in app.py** |
| Model benchmark dashboard | ✅ Yes | ✅ Partial — shows table + static image |
| Confusion matrix display | ✅ Mentioned | ❌ **Not in app.py** (generated as static PNGs during training) |
| Interactive EDA tab | ✅ Mentioned | ❌ **Not in app.py** (EDA runs during training, not in the web app) |
| Word clouds | ✅ Mentioned | ❌ **Not in app.py** (generated as static PNGs) |

> [!WARNING]
> The README mentions batch CSV processing, confusion matrices, EDA tabs, and word clouds as part of the Streamlit app. **The actual `app.py` code does NOT implement these features.** These are either planned future features or the README is aspirational. Do NOT claim these in an interview unless you add them.

> **What you should remember for the interview:**
> - `@st.cache_resource` ensures models load once, not per page refresh
> - The app uses majority voting across all 5 models for the final sentiment
> - The prediction function in app.py is self-contained (not using SentimentPredictor class)
> - Be honest: the app does single prediction + model comparison table. Batch CSV and EDA tabs are NOT implemented.

---

## SECTION 13: CODEBASE WALKTHROUGH

### 13.1 File-by-File Analysis

---

#### [config.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/config.py) — Configuration Hub

**Purpose**: Centralizes ALL hyperparameters, file paths, and settings in one place. No magic numbers scattered across the codebase.

**Key sections**:
| Section | What It Contains |
|---|---|
| Project Paths | `PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, `OUTPUTS_DIR` |
| Dataset | `IMDB_URL`, `DATASET_CSV` |
| Feature Engineering | `TFIDF_MAX_FEATURES=10000`, `W2V_VECTOR_SIZE=100`, `MAX_SEQUENCE_LENGTH=200`, etc. |
| Training | `TEST_SIZE=0.2`, `BATCH_SIZE=64`, `EPOCHS=5`, `VALIDATION_SPLIT=0.1` |
| Model Save Paths | Paths for all 5 models + 3 feature extractors |
| Hyperparameter Grids | `LR_PARAM_GRID`, `XGB_PARAM_GRID`, `RF_PARAM_GRID` for GridSearchCV |

**Connects to**: Imported by `main.py` and `app.py` via `from config import *`

---

#### [main.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/main.py) — Pipeline Orchestrator

**Purpose**: The entry point. Runs the complete 9-step pipeline from data loading to prediction testing.

**Key function**: `main()` — single function that orchestrates everything sequentially

**Flow**:
```python
def main():
    # Step 1: load_imdb_dataset()
    # Step 2: preprocess_dataset()
    # Step 3: run_eda()
    # Step 4: split_dataset() + 3 feature engineering functions
    # Step 5: 5 model training functions
    # Step 6: evaluate_all_models()
    # Step 7: run_hyperparameter_tuning()
    # Step 8: Print saved model sizes
    # Step 9: SentimentPredictor.load_models() + test_predictions()
```

**Connects to**: Imports from ALL `src/` modules and `config.py`

---

#### [src/utils.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/utils.py) — Console Formatting

**Purpose**: Provides consistent formatted console output throughout the pipeline.

**Functions**:
| Function | Output Example |
|---|---|
| `print_header("STEP 1")` | `======== STEP 1 ========` |
| `print_subheader("OVERVIEW")` | `------- OVERVIEW -------` |
| `print_info("Loading...")` | `[INFO] Loading...` |
| `print_success("Done!")` | `[OK] Done!` |
| `print_metric("Accuracy", "0.89")` | `\|-- Accuracy: 0.89` |
| `print_banner()` | ASCII art project banner |

Also forces **UTF-8 output on Windows** (lines 10-15) to avoid encoding errors with special characters.

**Connects to**: Imported by every other `src/` module.

---

#### [src/data_loader.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/data_loader.py) — Data Ingestion

**Purpose**: Downloads, parses, and caches the IMDB dataset.

**Key functions**:
| Function | Input | Output | Purpose |
|---|---|---|---|
| `download_from_stanford()` | URL, data_dir | extracted path or None | Download + extract tar.gz |
| `load_reviews_from_directory()` | path to aclImdb/ | DataFrame | Parse 50K .txt files into DataFrame |
| `load_from_keras_fallback()` | — | DataFrame | Fallback: decode Keras integer sequences to text |
| `load_imdb_dataset()` | data_dir, url, csv_path | DataFrame with `text` + `sentiment` | Main entry: cache check → download → parse |
| `split_dataset()` | DataFrame, test_size, random_state | X_train, X_test, y_train, y_test | Stratified 80/20 split |

**Connects to**: Called by `main.py` Steps 1 and 4.

---

#### [src/preprocessing.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/preprocessing.py) — Text Cleaning

**Purpose**: 12-step NLP preprocessing with negation preservation.

**Key functions**:
| Function | Input | Output |
|---|---|---|
| `preprocess_text(text)` | Raw string | Cleaned string |
| `preprocess_dataset(df)` | DataFrame with `text` column | DataFrame with `cleaned_text`, `original_length`, `cleaned_length`, `word_count` |

**Connects to**: Called by `main.py` Step 2, `prediction.py` for inference, and `app.py` for web predictions.

---

#### [src/feature_engineering.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py) — Text-to-Numbers

**Purpose**: Three parallel pipelines converting cleaned text to numerical features.

**Key functions**:
| Function | Input | Output | Saves |
|---|---|---|---|
| `create_tfidf_features()` | X_train, X_test text | `(sparse_train, sparse_test, vectorizer)` | `tfidf_vectorizer.pkl` |
| `create_word2vec_features()` | X_train, X_test text | `(w2v_train, w2v_test, w2v_model)` | `word2vec_model.model` |
| `prepare_dl_sequences()` | X_train, X_test text | `(padded_train, padded_test, tokenizer)` | `dl_tokenizer.pkl` |

**Connects to**: Called by `main.py` Step 4. Saved objects loaded by `prediction.py` and `app.py`.

---

#### [src/models.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/models.py) — Model Definitions & Training

**Purpose**: Defines and trains all 5 models.

**Key functions**:
| Function | Input Features | Saves |
|---|---|---|
| `train_logistic_regression()` | TF-IDF sparse | `logistic_regression.pkl` |
| `train_xgboost()` | TF-IDF sparse | `xgboost.pkl` |
| `train_random_forest()` | Word2Vec dense | `random_forest.pkl` |
| `build_lstm_model()` | — (architecture only) | — |
| `train_lstm()` | Padded sequences | `lstm_model.keras` |
| `build_bilstm_model()` | — (architecture only) | — |
| `train_bilstm()` | Padded sequences | `bilstm_model.keras` |

**Connects to**: Called by `main.py` Step 5.

---

#### [src/evaluation.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/evaluation.py) — Metrics & Visualization

**Purpose**: Compute metrics, generate confusion matrices, produce comparison charts.

**Key functions**:
| Function | Input | Output |
|---|---|---|
| `evaluate_model()` | model_name, y_true, y_pred | dict with accuracy, precision, recall, F1 |
| `plot_confusion_matrix()` | y_true, y_pred, name, path | Saves PNG heatmap |
| `evaluate_all_models()` | dict of (y_true, y_pred) per model | DataFrame + comparison chart + CSV |
| `plot_training_history()` | Keras history object | Saves accuracy/loss curve PNGs |

**Connects to**: Called by `main.py` Step 6.

---

#### [src/tuning.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/tuning.py) — Hyperparameter Optimization

**Purpose**: GridSearchCV for the 3 ML models.

**Key functions**:
| Function | Input | Method | Output |
|---|---|---|---|
| `tune_logistic_regression()` | TF-IDF + labels | GridSearchCV (3-fold) | Best model + best CV F1 |
| `tune_xgboost()` | TF-IDF + labels | GridSearchCV (3-fold) | Best model + best CV F1 |
| `tune_random_forest()` | Word2Vec + labels | GridSearchCV (3-fold) | Best model + best CV F1 |
| `run_hyperparameter_tuning()` | All features + config | Orchestrates all 3 | dict of best models |

**Connects to**: Called by `main.py` Step 7. DL models are NOT tuned via GridSearch — they use EarlyStopping instead.

---

#### [src/prediction.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/prediction.py) — Inference Pipeline

**Purpose**: Unified prediction interface using saved models.

**Key class**: `SentimentPredictor`
- `__init__(config)` — stores model paths
- `load_models()` — loads all from disk
- `predict(text, model_name=None)` — preprocesses + runs through selected model(s)
- `_predict_tfidf()` — TF-IDF → predict_proba
- `_predict_w2v()` — Word2Vec average → predict_proba
- `_predict_dl()` — Tokenize → pad → model.predict

**Also**: `test_predictions(predictor)` — tests 6 sample reviews

**Connects to**: Called by `main.py` Step 9.

---

#### [src/eda.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/eda.py) — Exploratory Data Analysis

**Purpose**: Generates 7 visualizations as PNG files.

**Function**: `run_eda(df, output_dir)` — single function that produces:
1. Sentiment distribution bar chart
2. Review length distribution histogram (by sentiment)
3. Word count distribution histogram (by sentiment)
4. Top 20 positive words horizontal bar chart
5. Top 20 negative words horizontal bar chart
6. Positive word cloud
7. Negative word cloud

**Connects to**: Called by `main.py` Step 3. Uses `matplotlib`, `seaborn`, `wordcloud`.

---

#### [app.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/app.py) — Streamlit Web App

**Purpose**: Interactive web UI for real-time sentiment prediction.

**Not a module** — it's a standalone script run via `streamlit run app.py`.

**Connects to**: Imports from `config.py` and `src/preprocessing.py`. Loads saved models from `models/` directory.

---

### 13.2 Complete Dependency Flow

```
config.py ─────────────────────────────────────────────────────┐
    │                                                          │
    ▼                                                          ▼
main.py ──imports──► src/utils.py ◄── imported by ALL src/*    app.py
    │                                                          │
    ├──► src/data_loader.py                                    │
    │        uses: pandas, sklearn.train_test_split             │
    │                                                          │
    ├──► src/preprocessing.py ◄────────────────────────────────┤
    │        uses: re, string, nltk                            │
    │                                                          │
    ├──► src/eda.py                                            │
    │        uses: matplotlib, seaborn, wordcloud, Counter     │
    │                                                          │
    ├──► src/feature_engineering.py                             │
    │        uses: sklearn.TfidfVectorizer, gensim.Word2Vec,   │
    │              keras.Tokenizer, keras.pad_sequences         │
    │                                                          │
    ├──► src/models.py                                         │
    │        uses: sklearn.LogisticRegression,                  │
    │              sklearn.RandomForestClassifier,              │
    │              xgboost.XGBClassifier,                       │
    │              keras.Sequential, keras.LSTM, etc.           │
    │                                                          │
    ├──► src/evaluation.py                                     │
    │        uses: sklearn.metrics, matplotlib, seaborn         │
    │                                                          │
    ├──► src/tuning.py                                         │
    │        uses: sklearn.GridSearchCV                         │
    │                                                          │
    └──► src/prediction.py ◄───────── prediction logic ────────┘
             uses: keras.load_model, gensim.Word2Vec, joblib
```

> **What you should remember for the interview:**
> - The project follows separation of concerns: each file has ONE responsibility
> - `config.py` is the single source of truth for all settings
> - `preprocessing.py` is shared between training (main.py) and inference (prediction.py, app.py)
> - Data flows: raw data → data_loader → preprocessing → feature_engineering → models → evaluation
> - `app.py` is independent of `main.py` — it loads saved artifacts, doesn't retrain

---

## SECTION 14: HYPERPARAMETERS TABLE

### Complete Hyperparameter Reference

| Parameter | Value | File & Line | Controls | Why This Value | If Increased | If Decreased |
|---|---|---|---|---|---|---|
| **Dataset & Split** |
| `TEST_SIZE` | 0.2 | config.py:36 | Train/test ratio | Standard 80/20 split | Less training data, more reliable test metrics | More training data, less reliable test |
| `RANDOM_STATE` | 42 | config.py:37 | Reproducibility seed | Convention (arbitrary) | Different random split | Different random split |
| **TF-IDF** |
| `TFIDF_MAX_FEATURES` | 10,000 | config.py:25 | Vocabulary size | Balance between coverage and noise | More features, slower, captures rare words | Fewer features, faster, misses rare words |
| `ngram_range` | (1, 2) | feature_eng:30 | Unigrams + bigrams | Captures 2-word phrases like "not good" | (1,3) adds trigrams: slower, possibly overfit | (1,1) loses phrase context |
| `min_df` | 5 | feature_eng:31 | Min document frequency | Removes very rare words (noise) | Higher = more aggressive filtering | Lower = more noise kept |
| `max_df` | 0.95 | feature_eng:32 | Max document frequency | Removes near-universal words | Higher = keeps more common words | Lower = removes more common words |
| `sublinear_tf` | True | feature_eng:33 | Log scaling of TF | Prevents frequency domination | — (boolean) | False = linear scaling, common words dominate |
| **Word2Vec** |
| `W2V_VECTOR_SIZE` | 100 | config.py:26 | Embedding dimensions | Standard for medium corpora | More expressive but slower training | Less expressive, faster |
| `W2V_WINDOW` | 5 | config.py:27 | Context window | 5 words each side = reasonable context | Larger context, slower | Narrow context, misses relationships |
| `W2V_MIN_COUNT` | 2 | config.py:28 | Min word frequency | Ignore words appearing once | More filtering of rare words | Include all words (noisy) |
| `W2V_WORKERS` | 4 | config.py:29 | CPU parallelism | 4 cores for training | Faster if more cores available | Slower |
| `epochs` (W2V) | 15 | feature_eng:81 | Training passes | Sufficient for convergence | Marginal improvement, slower | Under-trained embeddings |
| `sg` | 0 | feature_eng:83 | CBOW vs Skip-Gram | CBOW = faster | 1 = Skip-Gram: better for rare words, slower | — |
| **Deep Learning Sequences** |
| `MAX_NUM_WORDS` | 10,000 | config.py:31 | Tokenizer vocab cap | Top 10K words | More vocabulary coverage | Smaller vocabulary, more OOV |
| `MAX_SEQUENCE_LENGTH` | 200 | config.py:30 | Sequence length cap | Balance memory vs. info | Longer sequences, more memory, more info | Shorter, faster, loses more info |
| **Logistic Regression** |
| `C` | 1.0 | models.py:35 | Inverse regularization | Moderate regularization | Less regularization → risk overfit | More regularization → risk underfit |
| `penalty` | 'l2' | models.py:36 | Regularization type | L2 = ridge, shrinks all weights | — | L1 = lasso, can zero out weights |
| `solver` | 'lbfgs' | models.py:37 | Optimization algorithm | Efficient for L2 | — | Other solvers for different penalties |
| `max_iter` | 1000 | models.py:38 | Max optimization steps | Ensure convergence | More steps, slower | May not converge |
| **XGBoost** |
| `n_estimators` | 200 | models.py:61 | Number of trees | Good balance | More trees, slower, slightly better | Fewer trees, faster, worse |
| `max_depth` | 6 | models.py:62 | Tree depth | Moderate complexity | Deeper = more complex, risk overfit | Shallower = simpler, underfit |
| `learning_rate` | 0.1 | models.py:63 | Step size per tree | Standard starting point | Larger steps, fewer trees needed, risk overfit | Smaller, needs more trees, better generalization |
| `subsample` | 0.8 | models.py:64 | Row sampling per tree | Prevents overfitting | 1.0 = no sampling | Lower = more regularization |
| `colsample_bytree` | 0.8 | models.py:65 | Column sampling | Prevents overfitting | 1.0 = all features | Lower = more randomness |
| **Random Forest** |
| `n_estimators` | 200 | models.py:91 | Number of trees | Good ensemble size | More trees = better but diminishing returns | Fewer = higher variance |
| `max_depth` | None | models.py:92 | Tree depth limit | Fully grown trees | Set value = constrained trees | — (already unlimited) |
| `min_samples_split` | 2 | models.py:93 | Min samples to split | Allows finest splits | Higher = less overfitting per tree | — (already minimum) |
| **Neural Networks** |
| `embedding_dim` | 128 | models.py:176 | Embedding vector size | Expressive word representations | Richer but more parameters | Less expressive |
| LSTM units | 128 | models.py:130 | Hidden state size | Moderate capacity | More memory capacity, slower | Less capacity |
| `SpatialDropout1D` rate | 0.2 | models.py:129 | Embedding dropout | Mild regularization | More regularization | Less regularization |
| `Dropout` (post-LSTM) | 0.3 | models.py:131 | Feature dropout | Moderate regularization | Stronger regularization | Risk overfitting |
| Dense units | 64 | models.py:132 | Hidden layer size | Bottleneck compression | More capacity | More compression |
| `Dropout` (post-Dense) | 0.2 | models.py:133 | Feature dropout | Light regularization | Stronger regularization | Risk overfitting |
| `BATCH_SIZE` | 64 | config.py:38 | Samples per gradient step | Common balanced choice | Faster, noisier gradients | Slower, smoother gradients |
| `EPOCHS` | 5 | config.py:39 | Max training passes | Low — relies on EarlyStopping | More training time, potentially better | Even less training |
| `VALIDATION_SPLIT` | 0.1 | config.py:40 | Validation holdout | 10% for monitoring | More reliable validation, less training data | Less reliable validation |
| EarlyStopping patience | 2 | models.py:187 | Epochs without improvement | Stops quickly | Trains longer | Stops sooner |
| **GridSearchCV** |
| `cv` | 3 | tuning.py:25 | Cross-validation folds | 3-fold for speed | More folds = more reliable, slower | Faster, less reliable |
| `scoring` | 'f1' | tuning.py:26 | Optimization metric | F1 balances precision/recall | — | 'accuracy' may be fine for balanced data |

> **What you should remember for the interview:**
> - Be able to explain any hyperparameter from this table if asked
> - Key trade-offs: `C` (regularization), `max_depth` (tree complexity), `EPOCHS` (training duration), `MAX_SEQUENCE_LENGTH` (information retention)
> - The 5-epoch limit is arguably too low for LSTMs and partly explains their poor performance

---

## SECTION 15: MATHEMATICAL FOUNDATION

### 15.1 TF-IDF Mathematics

**Term Frequency with sublinear scaling:**
```
TF(t, d) = 1 + log(count(t in d))    if count > 0
          = 0                          otherwise
```

**Inverse Document Frequency:**
```
IDF(t, D) = log((1 + N) / (1 + df(t))) + 1
```
Where N = total documents, df(t) = docs containing term t.

**Example**: "good" appears 3 times in document d. Corpus has 1000 docs, "good" in 200.
```
TF  = 1 + log(3) = 1 + 1.099 = 2.099
IDF = log((1+1000)/(1+200)) + 1 = log(4.98) + 1 = 1.605 + 1 = 2.605
TF-IDF = 2.099 × 2.605 = 5.468
```

---

### 15.2 Sigmoid Function

```
σ(z) = 1 / (1 + e^(-z))
```

**Numerical examples:**
```
σ(-3) = 1/(1+e³) = 1/(1+20.09) = 0.047   → ~5% (strong negative)
σ(-1) = 1/(1+e¹) = 1/(1+2.72)  = 0.269   → ~27%
σ(0)  = 1/(1+1)  = 0.5                    → 50% (decision boundary)
σ(1)  = 1/(1+e⁻¹)= 1/(1+0.37)  = 0.731   → ~73%
σ(3)  = 1/(1+e⁻³)= 1/(1+0.05)  = 0.953   → ~95% (strong positive)
```

**Properties:**
- Maps any real number to (0, 1)
- Monotonically increasing
- Symmetric: σ(-z) = 1 - σ(z)
- Derivative: σ'(z) = σ(z)(1 - σ(z))

---

### 15.3 Logistic Regression

**Model:**
```
P(y=1|x) = σ(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
         = σ(w·x + b)
```

**Example** with 3 TF-IDF features:
```
w = [2.5, -1.8, 3.1]  (learned weights)
b = -0.5               (bias)
x = [0.4, 0.6, 0.0]   (TF-IDF values for one review)

z = 2.5(0.4) + (-1.8)(0.6) + 3.1(0.0) + (-0.5)
  = 1.0 - 1.08 + 0 - 0.5
  = -0.58

P(positive) = σ(-0.58) = 1/(1+e^0.58) = 1/(1+1.786) = 0.359

Since 0.359 < 0.5 → Predict NEGATIVE
```

---

### 15.4 Binary Cross-Entropy Loss

```
L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Example 1**: True label y=1 (positive), prediction ŷ=0.9
```
L = -[1·log(0.9) + 0·log(0.1)]
  = -log(0.9)
  = -(-0.105)
  = 0.105  (low loss — correct and confident)
```

**Example 2**: True label y=1, prediction ŷ=0.2
```
L = -[1·log(0.2) + 0·log(0.8)]
  = -log(0.2)
  = 1.609  (high loss — wrong and confident)
```

**Example 3**: True label y=0 (negative), prediction ŷ=0.1
```
L = -[0·log(0.1) + 1·log(0.9)]
  = -log(0.9)
  = 0.105  (low loss — correct)
```

**For entire dataset**: Average loss across all N samples:
```
Cost = (1/N) × Σ L(yᵢ, ŷᵢ)
```

---

### 15.5 Gradient Descent

**Goal**: Find weights w that minimize the cost function.

**Update rule:**
```
w_new = w_old - α × ∂Cost/∂w
```

Where α = learning rate (step size)

**Intuition**: Imagine standing on a hill (high cost). The gradient tells you which direction goes uphill. You step in the OPPOSITE direction (downhill) to reduce cost.

**Example** with learning rate α = 0.01:
```
Current weight: w = 2.0
Gradient: ∂Cost/∂w = 4.5  (cost increases if w increases)

w_new = 2.0 - 0.01 × 4.5 = 2.0 - 0.045 = 1.955
```

The weight decreased because the gradient was positive (cost was increasing in the positive direction).

**Variants used in this project:**
- LBFGS (Logistic Regression): Quasi-Newton method, uses second-order information (faster convergence)
- Adam (Neural Networks): Adaptive learning rate per parameter + momentum

---

### 15.6 L2 Regularization

**Without regularization:**
```
minimize: Cost(w)
```

**With L2 regularization:**
```
minimize: Cost(w) + λ × ||w||²
        = Cost(w) + λ × (w₁² + w₂² + ... + wₙ²)
```

Where λ = 1/(2C) in scikit-learn. With C=1.0, λ=0.5.

**What it does:** Penalizes large weights. If a weight grows too large, the penalty term increases, pushing the optimizer to find smaller weights.

**Why it helps:** Large weights = model is very sensitive to specific features = overfitting. Small weights = smoother, more generalizable model.

**Example:**
```
Without L2: w_"movie" = 0.01, w_"John" = 15.3
(model memorized that actor "John" appears in training positive reviews)

With L2:    w_"movie" = 0.4, w_"John" = 1.2
(model gives moderate weight to many features, not extreme weight to one)
```

---

### 15.7 Decision Trees & Gini Impurity

**Gini Impurity:**
```
Gini(S) = 1 - Σᵢ pᵢ²
```

**Example**: Split a node of 100 reviews into two children:

```
Before split: 60 positive, 40 negative
Gini_parent = 1 - (0.6² + 0.4²) = 1 - (0.36 + 0.16) = 0.48

Split on feature_42 > 0.3:
  Left child:  50 positive, 10 negative (60 total)
  Right child: 10 positive, 30 negative (40 total)

Gini_left  = 1 - (50/60)² - (10/60)² = 1 - 0.694 - 0.028 = 0.278
Gini_right = 1 - (10/40)² - (30/40)² = 1 - 0.0625 - 0.5625 = 0.375

Weighted Gini = (60/100)×0.278 + (40/100)×0.375
             = 0.167 + 0.150 = 0.317

Gini reduction = 0.48 - 0.317 = 0.163  (good split!)
```

The tree picks the split that **maximizes Gini reduction**.

---

### 15.8 Random Forest Mathematics

**Prediction (classification):**
```
ŷ = mode(h₁(x), h₂(x), ..., h₂₀₀(x))
```

Where each hᵢ is a decision tree trained on a bootstrap sample.

**Probability estimate:**
```
P(y=1|x) = (1/200) × Σ I(hᵢ(x) = 1)
```

Where I() is the indicator function. If 160 out of 200 trees predict positive:
```
P(positive) = 160/200 = 0.80
```

---

### 15.9 Word2Vec Concept (Simplified)

**Training objective (CBOW):** Given context words, predict the center word.

```
Context: ["the", "movie", "really", "great"]  →  Predict: "was"
```

The network has:
- Input: average of context word vectors
- Hidden layer: 100 neurons (= embedding dimension)
- Output: softmax over entire vocabulary

After training, the **hidden layer weights** ARE the word vectors.

**Cosine Similarity** — measures angle between two vectors:
```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

**Example:**
```
v("good")  = [0.5, 0.8, 0.3]
v("great") = [0.4, 0.7, 0.4]
v("bad")   = [-0.5, -0.7, -0.2]

cos(good, great) = (0.5×0.4 + 0.8×0.7 + 0.3×0.4) / (√(0.98) × √(0.81))
                 = (0.20 + 0.56 + 0.12) / (0.99 × 0.90)
                 = 0.88 / 0.891
                 = 0.988  (very similar!)

cos(good, bad) = (0.5×(-0.5) + 0.8×(-0.7) + 0.3×(-0.2)) / (0.99 × 0.89)
               = (-0.25 - 0.56 - 0.06) / 0.881
               = -0.87 / 0.881
               = -0.988  (very dissimilar — opposite!)
```

---

### 15.10 Embeddings

An embedding is a **learned mapping** from a discrete symbol (word, category) to a continuous dense vector.

```
"movie" → [0.12, -0.34, 0.56, 0.78, ..., -0.23]  (128 floats)
```

**Why not one-hot?** With 10,000 words, one-hot encoding creates 10,000-dim sparse vectors. Embeddings compress this to 128 dimensions while encoding semantic relationships.

```
One-hot: "movie" = [0, 0, ..., 1, ..., 0, 0]  (10,000 dims, 1 non-zero)
Embedding: "movie" = [0.12, -0.34, ..., -0.23]  (128 dims, all non-zero)
```

---

### 15.11 LSTM Gates (Mathematical)

At each timestep t, given input xₜ (128-dim) and previous hidden state hₜ₋₁ (128-dim):

**Forget gate:**
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)     → (128,) values in [0,1]
```

**Input gate:**
```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)     → (128,) values in [0,1]
```

**Candidate cell state:**
```
c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  → (128,) values in [-1,1]
```

**Cell state update:**
```
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ        → (128,)
```
(⊙ = element-wise multiplication)

**Output gate:**
```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)     → (128,) values in [0,1]
```

**Hidden state:**
```
hₜ = oₜ ⊙ tanh(cₜ)                → (128,)
```

**Parameter count for one LSTM(128) with input_dim=128:**
```
Each gate has: W of shape (256, 128) and b of shape (128,)
  [hₜ₋₁, xₜ] is (128+128) = 256 concatenated

4 gates × (256×128 + 128) = 4 × (32,768 + 128) = 4 × 32,896 = 131,584 parameters
```

> **What you should remember for the interview:**
> - Be able to write the sigmoid formula and compute σ(0), σ(1), σ(-1) by heart
> - Know binary cross-entropy: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
> - Gini impurity: 1 - Σpᵢ² (0 = pure, 0.5 = maximally impure for binary)
> - LSTM cell state update: cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ (forget old + add new)
> - Cosine similarity: dot product / (norm × norm), range [-1, 1]
> - L2 regularization penalizes ||w||², keeping weights small


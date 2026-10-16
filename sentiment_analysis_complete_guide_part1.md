# 🎬 Advanced Sentiment Analysis System — Complete Interview Mastery Guide

# PART 1: Foundation (Sections 1–5)

---

## SECTION 1: PROJECT OVERVIEW

### What is the project?

This is an **end-to-end Natural Language Processing (NLP) system** that automatically determines whether a movie review expresses a **positive** or **negative** opinion. It's not just one model — it's a complete pipeline that compares **5 different models** (3 traditional Machine Learning + 2 Deep Learning) to find the best approach.

### What problem does it solve?

Imagine Netflix, Amazon, or IMDB receiving millions of reviews daily. Humans can't read them all. This system automatically classifies each review as positive or negative, enabling:
- Automated customer feedback analysis
- Product rating aggregation
- Brand sentiment monitoring

### What is the input?

**Raw, unstructured English text** — a movie review like:
> "This movie was absolutely terrible! The acting was wooden and the plot made no sense."

### What is the output?

A **binary classification**:
- `1` = **Positive** sentiment (the reviewer liked the movie)
- `0` = **Negative** sentiment (the reviewer disliked the movie)

Plus a **confidence score** (0.0 to 1.0) showing how certain the model is.

### What is the complete end-to-end workflow?

```
Raw IMDB Reviews (50,000)
    │
    ▼
[1] Data Loading (download from Stanford / load from cache)
    │
    ▼
[2] Text Preprocessing (12-step NLP pipeline with negation preservation)
    │
    ▼
[3] Exploratory Data Analysis (7 visualizations: distributions, word clouds)
    │
    ▼
[4] Train/Test Split (80% train, 20% test, stratified)
    │
    ▼
[5] Feature Engineering (3 parallel pipelines):
    ├── TF-IDF (sparse 10,000-dim vectors)  → feeds LR + XGBoost
    ├── Word2Vec (dense 100-dim vectors)    → feeds Random Forest
    └── Keras Tokenizer (200-len sequences) → feeds LSTM + BiLSTM
    │
    ▼
[6] Model Training (5 models trained)
    │
    ▼
[7] Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
    │
    ▼
[8] Hyperparameter Tuning (GridSearchCV for ML, EarlyStopping for DL)
    │
    ▼
[9] Prediction Pipeline Testing (6 sample reviews tested)
    │
    ▼
[10] Streamlit Web Application (real-time inference UI)
```

### Architecture — Simple explanation

Think of it as a **factory assembly line**:
1. **Raw material arrives** (raw review text)
2. **Cleaning station** (preprocessing removes HTML, URLs, normalizes text)
3. **Three translation machines** (TF-IDF, Word2Vec, Tokenizer — each converts text to numbers in different ways)
4. **Five quality inspectors** (5 models — each looks at the numbers differently)
5. **Final report** (which inspector was most accurate?)

### Architecture — Technical explanation

The system follows a **modular pipeline architecture** where each component is a separate Python module under `src/`. The pipeline is orchestrated by [main.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/main.py) which calls each module sequentially. Feature extraction is parallelized into three branches because different model families require different input representations:

| Feature Pipeline | Output Shape | Consuming Models |
|---|---|---|
| TF-IDF Vectorizer | `(N, 10000)` sparse matrix | Logistic Regression, XGBoost |
| Word2Vec + Doc Averaging | `(N, 100)` dense matrix | Random Forest |
| Keras Tokenizer + Padding | `(N, 200)` integer sequences | LSTM, BiLSTM |

All feature extractors are **fit only on training data** (preventing data leakage) and saved to disk for inference reuse.

> **What you should remember for the interview:**
> - This is a binary text classification system comparing 5 models across 3 feature representations
> - The architecture prevents data leakage by fitting feature extractors only on training data
> - The modular design separates concerns: data loading → preprocessing → features → models → evaluation → prediction
> - It includes both a training pipeline (main.py) and a serving pipeline (app.py)

---

## SECTION 2: DATASET

### Which dataset is used?

The **IMDB Large Movie Review Dataset** (also called "IMDB 50K"), created by Stanford AI Lab researcher **Andrew L. Maas** (published at ACL 2011).

### Why this dataset?

1. **Standard benchmark** — Used across hundreds of NLP papers, so results are comparable
2. **Perfect balance** — Exactly 25,000 positive + 25,000 negative = no class imbalance issues
3. **Real-world text** — Actual user-written reviews with HTML artifacts, slang, varying lengths
4. **Binary labels** — Clean positive/negative labels (ratings 1-4 = negative, 7-10 = positive)

### Number of records

**50,000 reviews total** (before deduplication)

### Input features

| Column | Description | Data Type |
|---|---|---|
| `text` | The raw review text | `string` |
| `sentiment` | Binary label | `int64` (0 or 1) |

After preprocessing, additional columns are added:
| Column | Description |
|---|---|
| `cleaned_text` | Review after full NLP preprocessing |
| `original_length` | Character count of original text |
| `cleaned_length` | Character count of cleaned text |
| `word_count` | Number of words in cleaned text |

### Target variable

`sentiment` — binary:
- `0` = **Negative** (reviewer disliked the movie)
- `1` = **Positive** (reviewer liked the movie)

### Train/Test split

- **80% train** = ~40,000 reviews
- **20% test** = ~10,000 reviews
- **Stratified split** — `stratify=df['sentiment']` ensures equal positive/negative ratio in both train and test sets
- `random_state=42` — makes the split reproducible

This is implemented in [split_dataset()](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/data_loader.py#L230-L248).

### How the data is loaded

The loading logic in [data_loader.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/data_loader.py) has a **three-level fallback**:

1. **Cache check** — If `data/imdb_reviews.csv` already exists, load from CSV (fastest)
2. **Primary source** — Download from `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`, extract, parse each `.txt` file
3. **Keras fallback** — If Stanford download fails, use `tensorflow.keras.datasets.imdb` and decode integer sequences back to text

The dataset directory structure after extraction:
```
aclImdb/
├── train/
│   ├── pos/  (12,500 .txt files)
│   └── neg/  (12,500 .txt files)
└── test/
    ├── pos/  (12,500 .txt files)
    └── neg/  (12,500 .txt files)
```

Each `.txt` file contains one review. The label comes from the folder name (`pos` → 1, `neg` → 0).

### Labels representation

Labels are stored as integers:
- `0` = Negative
- `1` = Positive

The code maps them to strings for display: `sentiment_map = {1: 'Positive', 0: 'Negative'}`

### Issues with the dataset

1. **HTML artifacts** — Reviews contain `<br />`, `<p>`, and other HTML tags (from web scraping)
2. **URLs** — Some reviews link to external sites
3. **Duplicates** — The code checks for and removes duplicates: `df.duplicated().sum()`
4. **Variable length** — Reviews range from a few words to several thousand words
5. **Empty after cleaning** — Some very short reviews may become empty strings after preprocessing (these are removed)

### Is the dataset balanced?

**Yes, perfectly balanced**: 25,000 positive + 25,000 negative. The stratified split maintains this balance in train/test sets. This means **accuracy is a fair metric** — you don't need to worry about a model always predicting the majority class.

### How data leakage is prevented

**IMPLEMENTED** in the actual code:

1. **Stratified train/test split happens BEFORE feature engineering** — See [main.py line 87](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/main.py#L87): `split_dataset()` is called before any feature creation
2. **TF-IDF: `fit_transform` on train, `transform` on test** — See [feature_engineering.py line 38-39](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py#L38-L39)
3. **Word2Vec: trained only on training tokens** — See [feature_engineering.py line 75](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py#L75): `sentences=train_tokens`
4. **Keras Tokenizer: `fit_on_texts` only on train** — See [feature_engineering.py line 132](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py#L132): `tokenizer.fit_on_texts(X_train)`

> **What you should remember for the interview:**
> - IMDB 50K, Stanford AI Lab, perfectly balanced (25K pos + 25K neg)
> - 80/20 stratified split
> - Three-level data loading: cache CSV → Stanford download → Keras fallback
> - Data leakage prevention: all feature extractors fit ONLY on training data
> - Dataset has real-world noise: HTML tags, URLs, variable lengths

---

## SECTION 3: COMPLETE DATA FLOW

Let's trace **exactly** what happens to this review:

> **"This movie was not good at all, but the acting was amazing."**

### Stage 1: Raw Text
```
"This movie was not good at all, but the acting was amazing."
```
This is what the user types into the Streamlit app or what's read from a file.

### Stage 2: Preprocessing (function [preprocess_text()](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/preprocessing.py#L55-L115))

**Step 1 — Lowercase:**
```
"this movie was not good at all, but the acting was amazing."
```

**Step 2 — Remove HTML tags** (no HTML here, unchanged):
```
"this movie was not good at all, but the acting was amazing."
```

**Step 3 — Remove URLs** (none here, unchanged)

**Step 4 — Remove emojis** (none here, unchanged)

**Step 5 — Remove punctuation:**
```
"this movie was not good at all but the acting was amazing"
```
(comma and period removed)

**Step 6 — Remove numbers** (none here, unchanged)

**Step 7 — Remove special characters** (none here, unchanged)

**Step 8 — Tokenize** (split by whitespace):
```python
["this", "movie", "was", "not", "good", "at", "all", "but", "the", "acting", "was", "amazing"]
```

**Step 9 — Stopword removal (with negation preservation):**

Normal NLTK stopwords would remove: "this", "was", "not", "at", "all", "but", "the", "was"

But because `_KEEP_WORDS` includes `"not"`, it is **preserved**:
```python
["movie", "not", "good", "acting", "amazing"]
```

**Step 10 — Lemmatization:**
```python
["movie", "not", "good", "acting", "amazing"]
```
(All words are already in base form. If we had "movies" → "movie", "amazed" → "amazed")

**Step 11 — Remove single-character tokens** (none here)

**Step 12 — Rejoin:**
```
"movie not good acting amazing"
```

### Stage 3: Feature Extraction (Three Parallel Paths)

#### Path A: TF-IDF
The `TfidfVectorizer.transform()` converts the cleaned text into a **sparse vector of 10,000 dimensions**:

```python
# Simplified example (actual vector has 10,000 elements)
[0, 0, ..., 0.34, ..., 0, 0.52, ..., 0, 0.41, ..., 0]
  ↑                  ↑         ↑              ↑
  (most dims = 0)   "movie"   "good"        "amazing"
```

The non-zero values are TF-IDF weights. The bigram "not good" may also have its own dimension.

#### Path B: Word2Vec
Each word is looked up in the trained Word2Vec model to get a 100-dimensional dense vector, then all word vectors are **averaged**:

```python
vec_movie   = [0.12, -0.34, 0.56, ...]   # 100 dims
vec_not     = [-0.23, 0.45, -0.11, ...]   # 100 dims
vec_good    = [0.67, 0.23, 0.89, ...]     # 100 dims
vec_acting  = [0.45, -0.12, 0.33, ...]    # 100 dims
vec_amazing = [0.78, 0.56, 0.91, ...]     # 100 dims

doc_vector = mean([vec_movie, vec_not, vec_good, vec_acting, vec_amazing])
           = [0.358, 0.156, 0.516, ...]   # 100 dims
```

Shape: `(1, 100)` — one row, 100 features.

#### Path C: Keras Tokenizer
Each word is mapped to an integer index from the vocabulary:

```python
"movie" → 42,  "not" → 15,  "good" → 27,  "acting" → 156,  "amazing" → 203
```

Sequence: `[42, 15, 27, 156, 203]`

Then **padded** to length 200 with zeros:
```python
[42, 15, 27, 156, 203, 0, 0, 0, ..., 0]   # 200 elements total
```

Shape: `(1, 200)` — one sequence of 200 integers.

### Stage 4: Model Prediction

| Model | Input | Prediction | Confidence |
|---|---|---|---|
| Logistic Regression | TF-IDF sparse vector `(1, 10000)` | 0 (Negative) | 73.2% |
| XGBoost | TF-IDF sparse vector `(1, 10000)` | 0 (Negative) | 68.5% |
| Random Forest | W2V dense vector `(1, 100)` | 0 (Negative) | 81.0% |
| LSTM | Padded sequence `(1, 200)` | 1 (Positive) | 54.2% |
| BiLSTM | Padded sequence `(1, 200)` | 0 (Negative) | 62.1% |

*(Note: These confidence values are illustrative. The actual values depend on the trained model weights.)*

### Stage 5: Final Output

The Streamlit app uses **majority vote** across all models:
- 4 models say Negative, 1 says Positive → **Final: NEGATIVE**
- Average confidence displayed

> **What you should remember for the interview:**
> - You should be able to trace ANY review from raw text → cleaned text → three different numerical representations → model predictions
> - The key insight: the SAME cleaned text is fed into THREE DIFFERENT feature pipelines, and each feeds different models
> - "not good" is preserved as two separate tokens because of negation-aware stopword removal
> - Word2Vec averages ALL word vectors into one document vector — this is called "document embedding by averaging"

---

## SECTION 4: PREPROCESSING

The preprocessing pipeline lives in [preprocessing.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/preprocessing.py). Let's go through every step.

### 4.1 Setup (Lines 1-52)

```python
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
```

- `re` — Python's regular expression library for pattern matching
- `string` — Provides `string.punctuation` (all punctuation characters)
- `nltk` — Natural Language Toolkit, the standard NLP library

**NLTK resource downloads (lines 19-20):**
```python
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)
```
- `stopwords` — List of common English words ("the", "is", "at", etc.)
- `wordnet` — Lexical database for lemmatization
- `omw-1.4` — Open Multilingual WordNet (required by newer NLTK versions)

### 4.2 Negation-Aware Stopword List (Lines 23-32)

```python
_stop_words = set(stopwords.words('english'))

_KEEP_WORDS = {
    'not', 'no', 'nor', 'never', 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'barely', 'scarcely', 'very', 'really',
    'too', 'quite', 'rather', 'most', 'more', 'few', 'little',
    'against', 'above', 'below',
}
_stop_words -= _KEEP_WORDS
```

**Why this is critical:**

The standard NLTK stopword list includes words like "not", "no", "never". If you remove them:

| Review | After Standard Removal | Sentiment Prediction |
|---|---|---|
| "This movie was **not** good" | "movie good" | ❌ **POSITIVE** (wrong!) |
| "I would **never** watch this again" | "would watch" | ❌ **POSITIVE** (wrong!) |
| "**Not** bad at all" | "bad" | ❌ **NEGATIVE** (wrong!) |

By subtracting `_KEEP_WORDS` from the stopword set, these critical words survive:

| Review | After Negation-Aware Removal | Sentiment Prediction |
|---|---|---|
| "This movie was **not** good" | "movie **not** good" | ✅ **NEGATIVE** (correct!) |
| "I would **never** watch this again" | "**never** watch" | ✅ **NEGATIVE** (correct!) |
| "**Not** bad at all" | "**not** bad" | ✅ **POSITIVE** (correct!) |

The set also preserves **intensity modifiers**: "very", "really", "too", "quite", "rather", "most", "more". These words affect sentiment strength: "very good" is more positive than just "good".

### 4.3 Precompiled Regex Patterns (Lines 37-52)

```python
_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_EMOJI_RE = re.compile("[" u"\U0001F600-\U0001F64F" ... "]+", flags=re.UNICODE)
_NUMBERS_RE = re.compile(r'\d+')
_SPECIAL_RE = re.compile(r'[^a-zA-Z\s]')
_SPACES_RE = re.compile(r'\s+')
```

**Why precompiled?** These patterns are applied to 50,000 reviews. Compiling once and reusing is much faster than recompiling for each review.

| Pattern | Matches | Example |
|---|---|---|
| `_HTML_RE` | Any HTML tag | `<br />`, `<b>text</b>` |
| `_URL_RE` | URLs starting with http/www | `https://example.com` |
| `_EMOJI_RE` | Unicode emoji characters | 😀, 🎬, 👍 |
| `_NUMBERS_RE` | Any digits | `3/10`, `2023` |
| `_SPECIAL_RE` | Anything that's not a letter or space | `$`, `@`, `#` |

### 4.4 The preprocess_text Function (Lines 55-115)

Each step and **what happens if you remove it**:

| Step | What It Does | If Removed |
|---|---|---|
| 1. `text.lower()` | Converts to lowercase | "Good" and "good" treated as different words; vocabulary doubles |
| 2. `_HTML_RE.sub(' ', text)` | Removes `<br/>`, `<p>` etc. | HTML tags become noise tokens in the vocabulary |
| 3. `_URL_RE.sub('', text)` | Removes URLs | URLs add irrelevant, unique tokens |
| 4. `_EMOJI_RE.sub('', text)` | Removes emojis | Emojis create unknown tokens for Word2Vec |
| 5. `text.translate(...)` | Removes punctuation | Punctuation creates unique tokens like "good!" vs "good" |
| 6. `_NUMBERS_RE.sub('', text)` | Removes digits | Numbers like "10" or "3/10" are uninformative for sentiment |
| 7. `_SPECIAL_RE.sub('', text)` | Removes remaining special chars | Characters like $ or @ add noise |
| 8. `text.split()` | Tokenizes by whitespace | — (required step) |
| 9. Stopword removal | Removes common words (preserving negation) | Feature vectors dominated by "the", "is", "a" |
| 10. Lemmatization | Reduces to base form | "running", "runs", "ran" treated as different words |
| 11. `len(t) > 1` | Removes single characters | Leftover single letters like "s", "t" add noise |
| 12. `' '.join(tokens)` | Rejoins into string | — (required for TF-IDF/W2V input) |

### 4.5 Dataset-Level Preprocessing (Lines 118-191)

The [preprocess_dataset()](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/preprocessing.py#L118-L191) function:

1. **Shows before/after examples** — Prints 4 example transformations so you can verify the pipeline works
2. **Applies in batches of 5,000** — For memory efficiency and progress tracking
3. **Adds derived columns**: `original_length`, `cleaned_length`, `word_count`
4. **Removes empty reviews** — If preprocessing leaves nothing (e.g., a review that was only emojis)

> **What you should remember for the interview:**
> - The preprocessing has **12 steps**, each serving a specific purpose
> - The **MOST important design decision** is negation preservation — this is what you should explain first and most confidently
> - Standard stopword removal + sentiment analysis = **guaranteed incorrect results** for negated sentences
> - The `_KEEP_WORDS` set has two categories: **negation words** (not, never, no) and **intensity words** (very, really, too)
> - Lemmatization is preferred over stemming because Word2Vec needs real dictionary words for lookup
> - Regex patterns are precompiled for performance (50K reviews × 6 regex operations = 300K regex calls)

---

## SECTION 5: FEATURE ENGINEERING

The feature engineering code is in [feature_engineering.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py). Three completely different approaches are used to convert text into numbers.

### 5.1 TF-IDF (Term Frequency — Inverse Document Frequency)

#### What it is

TF-IDF converts text into a **sparse numerical vector** where each dimension represents a word (or bigram), and the value indicates how **important** that word is to the specific document **relative to the entire corpus**.

#### The Formula

```
TF-IDF(word, document, corpus) = TF(word, document) × IDF(word, corpus)
```

**Term Frequency (TF)** — How often does the word appear in THIS document?

With **sublinear scaling** (used in this project):
```
TF(t, d) = 1 + log(count of t in d)    if count > 0
          = 0                            if count = 0
```

**Inverse Document Frequency (IDF)** — How rare is the word across ALL documents?
```
IDF(t, corpus) = log((1 + N) / (1 + df(t))) + 1
```
Where:
- `N` = total number of documents
- `df(t)` = number of documents containing term `t`

#### Small Numerical Example

Suppose we have 3 documents:
```
Doc1: "movie good acting good"
Doc2: "movie bad plot"
Doc3: "movie good bad"
```

For the word **"good"** in **Doc1**:

**TF**: "good" appears 2 times → `TF = 1 + log(2) = 1 + 0.693 = 1.693`

**IDF**: 3 total docs, "good" appears in 2 docs → `IDF = log((1+3)/(1+2)) + 1 = log(1.33) + 1 = 0.288 + 1 = 1.288`

**TF-IDF**: `1.693 × 1.288 = 2.181`

For the word **"movie"** in **Doc1**:

**TF**: "movie" appears 1 time → `TF = 1 + log(1) = 1 + 0 = 1.0`

**IDF**: 3 total docs, "movie" appears in ALL 3 → `IDF = log((1+3)/(1+3)) + 1 = log(1) + 1 = 0 + 1 = 1.0`

**TF-IDF**: `1.0 × 1.0 = 1.0`

Notice: "good" gets a HIGHER TF-IDF than "movie" because "good" is more discriminating — it doesn't appear everywhere.

#### Why sublinear TF is used (`sublinear_tf=True`)

Without sublinear: a word appearing 100 times gets weight 100. With sublinear: `1 + log(100) = 5.6`.

**Why?** A word appearing 100 times is important, but NOT 100x more important than one appearing once. Logarithmic scaling provides a more realistic weighting.

#### Why unigram + bigram (`ngram_range=(1, 2)`)

- **Unigrams**: individual words like "good", "bad", "amazing"
- **Bigrams**: two-word phrases like "not good", "very bad", "highly recommend"

Bigrams capture **local context** that unigrams miss. The phrase "not good" as a single feature directly encodes negative sentiment, whereas the unigram "good" alone suggests positive.

#### Why 10,000 features (`max_features=10000`)

- The full vocabulary could be 50,000+ unique tokens
- Most are extremely rare (appearing in <5 documents)
- `max_features=10000` keeps the top 10,000 by **term frequency**
- This reduces dimensionality and removes noise while keeping discriminative features
- `min_df=5` additionally removes words appearing in fewer than 5 documents
- `max_df=0.95` removes words appearing in >95% of documents (too common)

#### What the resulting matrix looks like

```python
tfidf_train.shape = (40000, 10000)  # 40K reviews × 10K features
```

This is a **scipy sparse matrix** (CSR format). Most values are 0 because each review only contains a tiny fraction of the 10,000 vocabulary words. Sparse format stores only non-zero values, saving massive amounts of memory.

Dense representation would need: `40,000 × 10,000 × 8 bytes = 3.2 GB`
Sparse representation: ~100 MB (since each review has ~50-200 non-zero features)

#### Actual implementation

```python
vectorizer = TfidfVectorizer(
    max_features=10000,      # Keep top 10K features
    ngram_range=(1, 2),       # Unigrams + bigrams
    min_df=5,                 # Word must appear in ≥5 docs
    max_df=0.95,              # Word must appear in ≤95% of docs
    sublinear_tf=True,        # Use 1+log(tf) instead of raw tf
    strip_accents='unicode',  # Remove accent marks (é → e)
)

tfidf_train = vectorizer.fit_transform(X_train)  # FIT + transform
tfidf_test = vectorizer.transform(X_test)          # ONLY transform
```

> [!IMPORTANT]
> `fit_transform` on training data, `transform` on test data — this is how data leakage is prevented.

---

### 5.2 Word2Vec

#### What Word2Vec is

Word2Vec is a neural network that learns to represent each word as a **dense vector of real numbers** (called an "embedding") such that words with similar meanings have similar vectors.

For example:
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

#### CBOW vs Skip-Gram

| CBOW (Continuous Bag of Words) | Skip-Gram |
|---|---|
| Predicts the **center word** from surrounding context words | Predicts **context words** from the center word |
| Input: ["the", "movie", "really", "great"] → Output: "was" | Input: "was" → Output: ["the", "movie", "really", "great"] |
| **Faster to train** | Better for rare words |
| **Used in this project** (`sg=0`) | Not used here (`sg=1` would enable it) |

#### How training works (simplified)

1. **Slide a window** across the text: `"the [movie was really] great"` (window=2 means 2 words each side)
2. Feed context words into a shallow neural network (1 hidden layer)
3. The network tries to predict the center word
4. Through backpropagation, the **hidden layer weights** become the word vectors
5. After training on millions of context windows, similar words end up with similar vectors

#### What a word vector represents

Each of the 100 dimensions doesn't have a human-interpretable meaning, but collectively they encode:
- Semantic similarity: "good" and "great" are close; "good" and "terrible" are far
- Syntactic relationships: verbs cluster together, nouns cluster together
- Analogies: king-man+woman ≈ queen

#### Configuration in this project

```python
w2v_model = Word2Vec(
    sentences=train_tokens,   # Only training data!
    vector_size=100,          # 100-dimensional vectors
    window=5,                 # 5 words to the left, 5 to the right
    min_count=2,              # Ignore words appearing < 2 times
    workers=4,                # Use 4 CPU cores
    epochs=15,                # 15 passes over the data
    seed=42,                  # Reproducibility
    sg=0,                     # CBOW architecture
)
```

| Parameter | Value | Explanation |
|---|---|---|
| `vector_size=100` | 100 | Each word → 100 floats. Balance between expressiveness and computation |
| `window=5` | 5 | Context window of 5 words each direction |
| `min_count=2` | 2 | Words appearing only once are noise — skip them |
| `workers=4` | 4 | Multi-threaded training |
| `epochs=15` | 15 | 15 full passes over the training corpus |
| `sg=0` | CBOW | Faster than Skip-Gram, good enough for a large dataset |

#### OOV (Out-of-Vocabulary) words

At prediction time, if a word wasn't seen during training:
```python
vecs = [model.wv[w] for w in tokens if w in model.wv]  # Skip unknown words
```
If ALL words are OOV:
```python
return np.zeros(size)  # Return a zero vector
```

#### Document vector by averaging — Mathematical explanation

Given a document with words w₁, w₂, ..., wₙ where each wᵢ has vector v(wᵢ) ∈ ℝ¹⁰⁰:

```
doc_vector = (1/n) × Σ v(wᵢ)  for all wᵢ in vocabulary
```

**Example** with 3-dimensional vectors (real ones are 100-dim):

```
"movie not good"

v("movie") = [0.5,  0.2, -0.1]
v("not")   = [-0.3, 0.4,  0.6]
v("good")  = [0.7, -0.1,  0.8]

doc_vector = mean = [(0.5-0.3+0.7)/3, (0.2+0.4-0.1)/3, (-0.1+0.6+0.8)/3]
           = [0.3,  0.167, 0.433]
```

This single vector `[0.3, 0.167, 0.433]` now represents the entire document in a 3-dimensional space (in reality, 100-dimensional).

The actual implementation:
```python
def _doc_vector(tokens, model, size):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(size)
```

**Output shapes:**
```python
w2v_train.shape = (40000, 100)  # 40K reviews × 100 features
w2v_test.shape  = (10000, 100)  # 10K reviews × 100 features
```

---

### 5.3 Keras Tokenizer / Sequences (for Deep Learning)

#### Why deep learning needs a different representation

LSTMs process **sequences** — they need to see words in order, one at a time. TF-IDF and Word2Vec document vectors lose word order. So we need integer sequences that preserve the original word ordering.

#### Vocabulary and word-to-index mapping

```python
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
```

This builds a dictionary like:
```python
{
    '<OOV>': 1,
    'movie': 2,
    'film': 3,
    'good': 4,
    'not': 5,
    'great': 6,
    ...
    # Up to ~80,000 unique words
}
```

But `num_words=10000` means only the **top 10,000 most frequent words** are kept. The 10,001st word and beyond get mapped to `1` (`<OOV>`).

#### Unknown words

The `oov_token='<OOV>'` parameter ensures that during prediction, any word not in the top 10,000 gets mapped to index `1` instead of being silently dropped.

#### Sequence conversion

```python
train_seq = tokenizer.texts_to_sequences(X_train)
```

Example:
```
"movie not good acting amazing"
→ [2, 5, 4, 156, 203]
```

#### Padding

```python
train_padded = pad_sequences(train_seq, maxlen=200, padding='post', truncating='post')
```

Neural networks need **fixed-size inputs**. Reviews vary in length (10 to 2000+ words), so:
- **Short reviews**: Padded with zeros at the end (`padding='post'`)
- **Long reviews**: Truncated to 200 tokens (`truncating='post'`)

Example of a short review:
```
[2, 5, 4, 156, 203, 0, 0, 0, 0, ..., 0]   ← 200 elements
```

Example of a long review (>200 words): only first 200 tokens kept, rest discarded.

#### Why 200 tokens maximum (`MAX_SEQUENCE_LENGTH = 200`)

- **GPU memory**: Longer sequences = more memory per batch
- **Training time**: Sequence length directly impacts LSTM computation time
- **Diminishing returns**: Most sentiment is expressed in the first 200 words
- **Trade-off**: Some long reviews lose information (this hurts LSTM/BiLSTM accuracy — we'll discuss in Section 10)

#### What the neural network actually receives

```
Input tensor shape: (batch_size, 200)

Example batch of 3 reviews:
[[ 42,  15,  27, 156, 203,   0,   0, ...,   0],   ← 200 ints
 [  5, 891,  42,   7,  33, 401,  22, ...,   0],   ← 200 ints
 [ 42,  27,   5, 156,  42,  33,  27, ..., 891]]   ← 200 ints
```

The Embedding layer then converts each integer to a 128-dimensional vector, so the actual input to the LSTM becomes `(batch_size, 200, 128)`.

> **What you should remember for the interview:**
> - Three feature pipelines serve different model types — this is a key architectural decision
> - TF-IDF: sparse, bag-of-words, captures word importance but loses order
> - Word2Vec: dense, semantic, captures meaning but averages away order
> - Keras sequences: preserves order but truncates at 200 tokens
> - All three are fit ONLY on training data
> - TF-IDF bigrams partially capture order ("not good" as one feature)
> - Word2Vec document averaging is mathematically just the mean of all word vectors
> - The 200-token limit is a real trade-off that affects LSTM performance


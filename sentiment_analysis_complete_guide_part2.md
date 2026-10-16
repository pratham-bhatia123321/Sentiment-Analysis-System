# 🎬 Advanced Sentiment Analysis System — Complete Interview Mastery Guide

# PART 2: Models & Evaluation (Sections 6–10)

---

## SECTION 6: MACHINE LEARNING MODELS

All model code lives in [models.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/models.py).

### 6.1 Logistic Regression (Model 1)

#### What it is — Simple

Logistic Regression is a **linear classifier** that draws a straight line (or hyperplane in high dimensions) to separate positive from negative reviews. Despite the word "regression," it's actually used for **classification**.

#### What it is — Technical

Logistic Regression models the probability of the positive class as:

```
P(y=1 | x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))
```

Where:
- `x` = the input feature vector (10,000-dim TF-IDF vector)
- `w` = weight vector (10,000 learned weights, one per feature)
- `b` = bias term
- `σ` = sigmoid function

#### Why it works well for text

1. **High-dimensional sparse data** — TF-IDF creates 10,000-dimensional vectors where most values are zero. In such high dimensions, data is often **linearly separable** (a hyperplane can separate positive from negative)
2. **Fast training** — The LBFGS solver converges quickly on convex loss functions
3. **Interpretable** — You can look at which words have the highest positive/negative weights

#### Sigmoid function

```
σ(z) = 1 / (1 + e^(-z))
```

| z value | σ(z) | Interpretation |
|---|---|---|
| -10 | 0.00005 | Very confident NEGATIVE |
| -2 | 0.119 | Likely negative |
| 0 | 0.500 | Coin flip — decision boundary |
| 2 | 0.881 | Likely positive |
| 10 | 0.99995 | Very confident POSITIVE |

The sigmoid squashes any real number into the range (0, 1), making it interpretable as a probability.

#### Decision boundary

The decision boundary is where `σ(w·x + b) = 0.5`, which means `w·x + b = 0`.

- If `w·x + b > 0` → predict Positive (class 1)
- If `w·x + b < 0` → predict Negative (class 0)

#### Regularization (L2)

```python
model = LogisticRegression(
    C=1.0,           # Inverse regularization strength
    penalty='l2',    # L2 regularization
    solver='lbfgs',  # Optimization algorithm
    max_iter=1000,   # Max iterations for convergence
    random_state=42,
    n_jobs=-1,       # Use all CPU cores
)
```

**L2 regularization** adds a penalty term to prevent overfitting:

```
Loss = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] + (1/2C) × ||w||²
```

- `C=1.0` — Controls penalty strength. `C` is the **inverse** of regularization strength:
  - Large C (e.g., 10) → weak regularization → model fits training data closely (risk of overfitting)
  - Small C (e.g., 0.01) → strong regularization → simpler model (risk of underfitting)
  - `C=1.0` → moderate balance

- `solver='lbfgs'` — Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm. A quasi-Newton method efficient for L2-regularized problems with many features.

#### Training and prediction

```python
model.fit(X_train, y_train)           # Learn w and b
prediction = model.predict(X_test)     # Apply threshold at 0.5
probabilities = model.predict_proba(X_test)  # Get [P(neg), P(pos)]
```

The `coef_` attribute has shape `(1, 10000)` — one weight per TF-IDF feature.

---

### 6.2 XGBoost (Model 2)

#### What it is

XGBoost (eXtreme Gradient Boosting) is an **ensemble method** that builds many small decision trees **sequentially**, where each new tree tries to correct the mistakes of the previous ones.

#### How boosting works — Step by step

1. **Tree 1**: Tries to classify all reviews. Gets some wrong.
2. **Tree 2**: Focuses MORE on the reviews Tree 1 got wrong. Builds a tree to fix those errors.
3. **Tree 3**: Focuses on what Trees 1+2 still get wrong.
4. ... continues for 200 trees (`n_estimators=200`)
5. **Final prediction**: Weighted sum of all 200 tree predictions, passed through sigmoid.

```
ŷ = σ(f₁(x) + f₂(x) + f₃(x) + ... + f₂₀₀(x))
```

Each `fᵢ(x)` is a shallow decision tree that outputs a small correction.

#### Configuration

```python
model = XGBClassifier(
    n_estimators=200,       # 200 boosting rounds (trees)
    max_depth=6,            # Each tree has max 6 levels
    learning_rate=0.1,      # Shrinkage — how much each tree contributes
    subsample=0.8,          # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,   # Use 80% of features per tree
    random_state=42,
    eval_metric='logloss',  # Binary cross-entropy for evaluation
    n_jobs=-1,
    verbosity=0,
)
```

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators=200` | 200 trees | More trees = better performance but slower |
| `max_depth=6` | 6 levels | Controls tree complexity. Deep trees overfit |
| `learning_rate=0.1` | 0.1 | Each tree's contribution is scaled by 0.1. Smaller = more trees needed but usually better |
| `subsample=0.8` | 80% | Random sampling of training data per tree (like bagging) |
| `colsample_bytree=0.8` | 80% | Random sampling of features per tree |

#### Why XGBoost was tested

XGBoost is a state-of-the-art algorithm for structured/tabular data. It's commonly competitive with or superior to deep learning for non-image/non-sequence problems. Testing it here benchmarks a strong tree-based model against linear (LR) and neural (LSTM) approaches.

---

### 6.3 Random Forest (Model 3)

#### How Random Forest works

Random Forest is an **ensemble of independent decision trees** where each tree is built on a different random subset of data and features. The final prediction is the **majority vote** across all trees.

```
Final prediction = mode(Tree₁(x), Tree₂(x), ..., Tree₂₀₀(x))
```

#### Decision Trees — How they split

A decision tree asks yes/no questions about features:

```
Is feature_42 (word2vec dim 42) > 0.15?
├── YES: Is feature_7 > -0.23?
│        ├── YES: POSITIVE (95% confidence)
│        └── NO:  NEGATIVE (72% confidence)
└── NO:  Is feature_91 > 0.44?
         ├── YES: POSITIVE (61% confidence)
         └── NO:  NEGATIVE (88% confidence)
```

#### Bagging (Bootstrap Aggregating)

Each of the 200 trees is trained on a **random sample with replacement** from the training data. This means each tree sees a slightly different dataset:
- Tree 1: samples {1, 3, 3, 5, 7, 7, 9, ...}
- Tree 2: samples {2, 2, 4, 5, 6, 8, 8, ...}
- Tree 3: samples {1, 1, 3, 6, 7, 9, 9, ...}

This **reduces variance** and prevents overfitting.

#### Random feature selection

At each split, the tree doesn't look at ALL 100 features (Word2Vec dimensions). It randomly selects √100 ≈ 10 features and picks the best split among those. This **decorrelates** the trees — each tree learns different patterns.

#### Gini Impurity

The decision tree picks the best split using **Gini impurity**:

```
Gini(S) = 1 - Σ pᵢ²
```

Where pᵢ is the proportion of class i in set S.

**Example**: A node with 70 positive and 30 negative reviews:
```
p_pos = 70/100 = 0.7
p_neg = 30/100 = 0.3
Gini = 1 - (0.7² + 0.3²) = 1 - (0.49 + 0.09) = 0.42
```

A **pure** node (all one class): `Gini = 1 - 1² = 0` (perfect)
A **maximally impure** node (50/50): `Gini = 1 - (0.5² + 0.5²) = 0.5` (worst)

The tree picks the split that **reduces Gini the most**.

#### Configuration

```python
model = RandomForestClassifier(
    n_estimators=200,      # 200 trees
    max_depth=None,        # Trees grow until pure or min_samples_split
    min_samples_split=2,   # A node needs at least 2 samples to split
    random_state=42,
    n_jobs=-1,             # Use all CPU cores
)
```

`max_depth=None` means trees grow **fully** — they keep splitting until every leaf is pure or has only 2 samples. This makes each individual tree overfit, but the ensemble averaging fixes this.

#### How it receives Word2Vec and produces sentiment

```
Input:  doc_vector of shape (1, 100)  — 100 dense float features
        ↓
        Each of 200 trees classifies the vector independently
        ↓
        Majority vote → final class (0 or 1)
        ↓
        predict_proba gives [P(neg), P(pos)] = fraction of trees voting each way
```

> **What you should remember for the interview:**
> - LR = linear model, good for high-dimensional sparse data (TF-IDF)
> - XGBoost = sequential trees, each corrects previous mistakes (boosting)
> - RF = parallel independent trees, majority vote (bagging)
> - LR decision boundary: σ(w·x+b) = 0.5
> - Random Forest with max_depth=None: fully grown trees, relies on ensemble averaging to prevent overfitting
> - Gini impurity measures node purity; splits minimize impurity
> - Key difference: boosting (XGBoost) is sequential, bagging (RF) is parallel

---

## SECTION 7: DEEP LEARNING MODELS

### 7.1 Architecture Overview

Both LSTM and BiLSTM follow the same structure, differing only in the recurrent layer:

```
Input: Integer sequence (batch_size, 200)
  ↓
Layer 1: Embedding         → (batch_size, 200, 128)
  ↓
Layer 2: SpatialDropout1D  → (batch_size, 200, 128)
  ↓
Layer 3: LSTM or BiLSTM    → (batch_size, 128) or (batch_size, 256)
  ↓
Layer 4: Dropout(0.3)      → same shape
  ↓
Layer 5: Dense(64, relu)   → (batch_size, 64)
  ↓
Layer 6: Dropout(0.2)      → (batch_size, 64)
  ↓
Layer 7: Dense(1, sigmoid) → (batch_size, 1)
```

### 7.2 Layer-by-Layer Explanation

#### Layer 1: Embedding

```python
Embedding(vocab_size, 128, input_length=200)
```

**What enters**: Integer sequence `[42, 15, 27, 156, 203, 0, 0, ..., 0]` — shape `(batch, 200)`

**What happens internally**: Each integer index is used as a lookup key into an **embedding matrix** of shape `(vocab_size, 128)`. Row 42 of the matrix becomes the vector for word 42.

```
Embedding matrix (10000 × 128):
Row 0:   [0.00, 0.00, ..., 0.00]   ← padding token
Row 1:   [0.12, -0.34, ..., 0.56]  ← <OOV> token
Row 2:   [-0.23, 0.45, ..., -0.11] ← "movie"
...
Row 42:  [0.67, 0.23, ..., 0.89]   ← some word
```

**What comes out**: `(batch, 200, 128)` — each of 200 positions now has a 128-dimensional vector

**Why**: Neural networks can't process integers directly. The embedding layer converts each word index into a learnable dense vector. Unlike Word2Vec (which is pretrained), this embedding is **trained from scratch** along with the rest of the model.

**Why 128 dimensions** (`embedding_dim=128`): Larger than Word2Vec's 100 because the LSTM needs richer representations since it's learning embeddings and task simultaneously. 128 is a common power-of-2 choice for GPU efficiency.

#### Layer 2: SpatialDropout1D(0.2)

**What enters**: `(batch, 200, 128)` — the embedding matrix for each review

**What happens internally**: Randomly drops **entire embedding dimensions** across all 200 time steps. If dimension 42 is dropped, it's zeroed out for ALL words in the sequence.

```
Normal Dropout:         SpatialDropout1D:
[0.5  0.3  0    0.7]   [0    0.3  0.8  0]      ← dim 0 and 3 dropped
[0.2  0    0.4  0.1]   [0    0.1  0.4  0]      ← same dims dropped
[0    0.6  0.9  0.3]   [0    0.6  0.9  0]      ← for ALL timesteps
```

**What comes out**: `(batch, 200, 128)` — same shape, but some entire channels are zeroed

**Why**: Regular dropout zeroes random individual elements, which doesn't prevent adjacent word embeddings from co-adapting. SpatialDropout1D drops **entire feature channels**, forcing the network to not rely on any single embedding dimension. This is more effective for NLP embeddings.

**Rate 0.2**: 20% of the 128 dimensions are dropped during each training step.

#### Layer 3a: LSTM(128)

**What enters**: `(batch, 200, 128)` — a sequence of 200 vectors, each 128-dimensional

**What happens internally**: The LSTM processes the sequence **one timestep at a time**, from left to right. At each timestep t, it receives:
- The current word vector `x_t` (128-dim)
- The previous hidden state `h_(t-1)` (128-dim)
- The previous cell state `c_(t-1)` (128-dim)

And produces:
- Updated hidden state `h_t` (128-dim)
- Updated cell state `c_t` (128-dim)

##### LSTM Gates — The Core Mechanism

The LSTM has **4 gates** (actually 3 gates + 1 candidate):

**1. Forget Gate** — "What old information should I throw away?"
```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)
```
Outputs values between 0 (forget completely) and 1 (keep fully).

Example: After reading "The movie was great but...", the forget gate might decide to forget the "great" signal because "but" often signals a reversal.

**2. Input Gate** — "What new information should I store?"
```
i_t = σ(W_i · [h_(t-1), x_t] + b_i)
```
Decides which values of the cell state to update.

**3. Candidate Cell State** — "What is the new information?"
```
c̃_t = tanh(W_c · [h_(t-1), x_t] + b_c)
```
Creates new candidate values to potentially add to the cell state.

**4. Cell State Update** — Combine forget + input:
```
c_t = f_t ⊙ c_(t-1) + i_t ⊙ c̃_t
```
Where ⊙ means element-wise multiplication.

**5. Output Gate** — "What part of the cell state should I output?"
```
o_t = σ(W_o · [h_(t-1), x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

##### Why LSTM is useful for text

- **Long-term dependencies**: The cell state `c_t` can carry information across many timesteps. If "not" appears at position 5 and "good" at position 50, the cell state can remember the negation.
- **Selective memory**: Gates learn WHEN to remember and WHEN to forget
- **Vanishing gradient solved**: The cell state's additive update (not multiplicative) allows gradients to flow back through long sequences

**What comes out**: With `return_sequences=False` (default), only the **final hidden state** `h_200` is returned — shape `(batch, 128)`. This 128-dim vector summarizes the entire 200-word sequence.

#### Layer 3b: Bidirectional(LSTM(128)) — BiLSTM alternative

```python
Bidirectional(LSTM(128))
```

**What enters**: `(batch, 200, 128)`

**What happens internally**: TWO separate LSTM networks process the sequence:
- **Forward LSTM**: reads left-to-right: word₁ → word₂ → ... → word₂₀₀
- **Backward LSTM**: reads right-to-left: word₂₀₀ → word₁₉₉ → ... → word₁

**Why bidirectional?**

Consider: "Not only was the plot weak, but the acting was also terrible."

- Forward LSTM at "weak": has seen "not only" → knows negation/emphasis is happening
- Backward LSTM at "weak": has seen "terrible" and "also" → knows the overall sentiment is negative

The backward pass lets the model use **future context** to understand current words.

**How they're combined**: By default, Keras **concatenates** the forward and backward final hidden states:

```
output = [h_forward_200 ; h_backward_1]
       = (batch, 128 + 128) = (batch, 256)
```

So BiLSTM output is **256-dim** (double the LSTM's 128-dim).

#### Layer 4: Dropout(0.3)

**What enters**: `(batch, 128)` for LSTM or `(batch, 256)` for BiLSTM

**What happens**: Randomly zeros 30% of elements during training. During inference (prediction), all elements are kept but scaled by 0.7.

**Why**: Prevents overfitting by forcing the network to not rely on any single neuron. The 30% rate is higher than normal because the LSTM output is a high-capacity representation that can easily memorize training data.

#### Layer 5: Dense(64, activation='relu')

**What enters**: `(batch, 128)` or `(batch, 256)`

**What happens**: Standard fully-connected layer: `output = relu(W·input + b)`

```
ReLU(z) = max(0, z)
```

| Input z | Output ReLU(z) |
|---|---|
| -5 | 0 |
| -0.1 | 0 |
| 0 | 0 |
| 0.1 | 0.1 |
| 5 | 5 |

**What comes out**: `(batch, 64)` — a compressed representation

**Why ReLU**: Prevents vanishing gradients (unlike sigmoid/tanh in deep networks), allows the model to learn nonlinear decision boundaries, and is computationally cheap.

**Why 64 units**: Acts as a bottleneck — compresses 128 (or 256) features into 64, forcing the model to learn the most important patterns. Like a funnel before the final classification.

#### Layer 6: Dropout(0.2)

Another dropout layer, 20% rate. Lighter than the previous 30% because the representation is already more compact.

#### Layer 7: Dense(1, activation='sigmoid')

**What enters**: `(batch, 64)`

**What happens**: `output = σ(W·input + b)` where `W` is a `(64, 1)` weight matrix

**What comes out**: `(batch, 1)` — a single probability value between 0 and 1

- Output > 0.5 → Positive
- Output ≤ 0.5 → Negative

**Why sigmoid and not softmax?** For binary classification, sigmoid on a single output neuron is equivalent to softmax on two neurons, but simpler. The output IS the probability of the positive class.

### 7.3 Training Configuration

#### Binary Cross-Entropy Loss

```
Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

| True label y | Prediction ŷ | Loss |
|---|---|---|
| 1 (positive) | 0.9 | -log(0.9) = 0.105 (low — good!) |
| 1 (positive) | 0.1 | -log(0.1) = 2.303 (high — bad!) |
| 0 (negative) | 0.1 | -log(0.9) = 0.105 (low — good!) |
| 0 (negative) | 0.9 | -log(0.1) = 2.303 (high — bad!) |

The loss **heavily penalizes confident wrong predictions**.

#### Adam Optimizer

Adam = Adaptive Moment Estimation. Combines:
- **Momentum** (SGD with momentum): Keeps a running average of past gradients to smooth out updates
- **RMSProp**: Adapts learning rate per parameter based on gradient magnitude

Default learning rate: `0.001`

#### EarlyStopping

```python
EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
```

- Monitors **validation loss** after each epoch
- If val_loss doesn't improve for **2 consecutive epochs**, training stops
- Restores the model weights from the **best epoch** (not the last)

This prevents overfitting — training stops when the model starts memorizing training data instead of learning general patterns.

#### ModelCheckpoint

```python
ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
```

Saves the model weights whenever **validation accuracy** improves. Only keeps the best version.

> **What you should remember for the interview:**
> - LSTM has 3 gates (forget, input, output) + candidate cell state
> - Cell state is the "long-term memory" — carries info across timesteps via additive updates
> - Hidden state is the "short-term memory" / output at each step
> - BiLSTM reads both directions and concatenates → double the hidden units
> - Embedding layer: integer → dense vector (learned during training)
> - SpatialDropout1D drops entire channels, not individual elements
> - Binary cross-entropy is the correct loss for binary classification with sigmoid output
> - EarlyStopping prevents overfitting by monitoring validation loss

> [!IMPORTANT]
> **Code vs README discrepancy detected**: The README states `patience=3` for EarlyStopping, but the actual code in [models.py line 187](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/models.py#L187) uses `patience=2`. The **code is the ground truth**. In an interview, say `patience=2`.

---

## SECTION 8: MODEL TRAINING

### 8.1 The Training Flow

The complete training is orchestrated in [main.py](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/main.py) Steps 4-5.

```
Step 4: Feature Engineering
├── split_dataset(df) → X_train, X_test, y_train, y_test
├── create_tfidf_features(X_train, X_test) → tfidf_train, tfidf_test
├── create_word2vec_features(X_train, X_test) → w2v_train, w2v_test
└── prepare_dl_sequences(X_train, X_test) → dl_train, dl_test

Step 5: Model Training
├── train_logistic_regression(tfidf_train, y_train)
├── train_xgboost(tfidf_train, y_train)
├── train_random_forest(w2v_train, y_train)
├── train_lstm(dl_train, y_train)
└── train_bilstm(dl_train, y_train)
```

### 8.2 FIT vs TRANSFORM — Critical Distinction

This is one of the **most commonly asked interview questions**.

| Object | FIT on training data | TRANSFORM on test data |
|---|---|---|
| TF-IDF Vectorizer | `vectorizer.fit_transform(X_train)` — learns vocabulary + IDF weights | `vectorizer.transform(X_test)` — uses existing vocabulary |
| Word2Vec Model | `Word2Vec(sentences=train_tokens)` — learns word vectors | Test tokens are looked up in the learned vocabulary |
| Keras Tokenizer | `tokenizer.fit_on_texts(X_train)` — builds word-to-index mapping | `tokenizer.texts_to_sequences(X_test)` — uses existing mapping |
| All ML Models | `model.fit(X_train, y_train)` — learns weights/trees | `model.predict(X_test)` — applies learned weights |
| All DL Models | `model.fit(X_train, y_train)` — learns neural network weights | `model.predict(X_test)` — forward pass only |

**If you fit on test data** (data leakage):
- TF-IDF: Test vocabulary would influence IDF weights → artificially high accuracy
- Word2Vec: Test words would be part of the embedding space → model "knows" test vocabulary
- Models: Would directly memorize test answers → meaningless metrics

### 8.3 Training Parameters for Deep Learning

```python
history = model.fit(
    X_train, y_train,
    batch_size=64,         # 64 sequences per gradient update
    epochs=5,              # Maximum 5 passes through training data
    validation_split=0.1,  # 10% of training data used for validation
    callbacks=[EarlyStopping, ModelCheckpoint],
)
```

| Parameter | Value | Explanation |
|---|---|---|
| `batch_size=64` | 64 | Number of samples processed before updating weights. Smaller = noisier gradients but better generalization. 64 is a common balanced choice |
| `epochs=5` | 5 | Maximum number of full passes through training data. EarlyStopping may stop earlier |
| `validation_split=0.1` | 10% | From the ~40K training samples, 4K are held out for validation monitoring. **These are NOT the test set** |

### 8.4 Avoiding Overfitting — Techniques Used

| Technique | Where Applied | How It Helps |
|---|---|---|
| L2 regularization | Logistic Regression (`penalty='l2'`) | Penalizes large weights |
| Subsample/colsample | XGBoost (both at 0.8) | Each tree sees different data/features |
| Ensemble averaging | Random Forest (200 trees) | Averages out individual tree overfitting |
| EarlyStopping | LSTM, BiLSTM | Stops training when validation loss plateaus |
| Dropout | LSTM, BiLSTM (0.2 and 0.3) | Randomly zeros neurons during training |
| SpatialDropout1D | LSTM, BiLSTM (0.2) | Drops entire embedding channels |

### 8.5 Preventing Data Leakage — Complete List

**IMPLEMENTED** — verified in code:

1. ✅ Split happens BEFORE feature engineering ([main.py line 87](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/main.py#L87))
2. ✅ TF-IDF: `fit_transform` on train, `transform` on test ([feature_engineering.py line 38-39](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py#L38-L39))
3. ✅ Word2Vec: trained only on `train_tokens` ([feature_engineering.py line 75](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py#L75))
4. ✅ Keras tokenizer: `fit_on_texts(X_train)` only ([feature_engineering.py line 132](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/feature_engineering.py#L132))
5. ✅ Stratified split preserves class balance ([data_loader.py line 239](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/src/data_loader.py#L239))

**POTENTIAL CONCERN** (not a leak, but worth mentioning):
- Preprocessing (`preprocess_dataset`) is applied to the **entire DataFrame** before splitting. This is fine because preprocessing is a **deterministic, stateless transformation** — it doesn't learn any statistics from the data. Each review is processed independently.

> **What you should remember for the interview:**
> - FIT = learn parameters from data. TRANSFORM = apply learned parameters.
> - NEVER fit on test data. This is data leakage.
> - EarlyStopping monitors VALIDATION loss (not training loss) — training loss always decreases
> - Preprocessing before split is OK because it's stateless (no learned statistics)
> - validation_split=0.1 creates a THIRD subset from training data — not the test set

---

## SECTION 9: MODEL EVALUATION

### 9.1 Metrics Explained with Numerical Examples

#### Confusion Matrix

For a binary classifier predicting Positive/Negative:

```
                    Predicted
                 Neg    Pos
Actual  Neg    [TN]   [FP]
        Pos    [FN]   [TP]
```

- **TP** (True Positive): Actually positive, predicted positive ✅
- **TN** (True Negative): Actually negative, predicted negative ✅
- **FP** (False Positive): Actually negative, predicted positive ❌ (Type I error)
- **FN** (False Negative): Actually positive, predicted negative ❌ (Type II error)

**Example**: Out of 100 reviews (50 positive, 50 negative):

```
                  Predicted
                Neg    Pos
Actual  Neg    [45]   [ 5]    → 45 correct negatives, 5 wrong
        Pos    [ 8]   [42]    → 42 correct positives, 8 missed
```

#### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (42 + 45) / 100 = 87/100 = 0.87 = 87%
```

"What fraction of ALL predictions were correct?"

#### Precision

```
Precision = TP / (TP + FP) = 42 / (42 + 5) = 42/47 = 0.894 = 89.4%
```

"Of all reviews I predicted as POSITIVE, what fraction actually were positive?"

High precision = few false alarms.

#### Recall (Sensitivity)

```
Recall = TP / (TP + FN) = 42 / (42 + 8) = 42/50 = 0.84 = 84%
```

"Of all ACTUALLY POSITIVE reviews, what fraction did I catch?"

High recall = few missed positives.

#### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.894 × 0.84) / (0.894 + 0.84)
   = 2 × 0.751 / 1.734
   = 0.866 = 86.6%
```

F1 is the **harmonic mean** of precision and recall. It penalizes models that sacrifice one for the other. A model with 100% precision but 10% recall gets F1 = 18.2% (not 55%).

### 9.2 Actual Project Results

From the [model_results.csv](file:///C:/Users/Garvita/.gemini/antigravity-ide/scratch/Sentiment-Analysis-System/outputs/model_results.csv):

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| TF-IDF + LR | **89.00%** | 88.37% | 89.91% | **89.13%** |
| BiLSTM | 87.45% | 89.38% | 85.09% | 87.18% |
| TF-IDF + XGBoost | 84.84% | 82.85% | 88.02% | 85.36% |
| W2V + Random Forest | 83.92% | 82.72% | 85.90% | 84.28% |
| LSTM | 53.30% | 51.89% | 95.66% | 67.28% |

> [!WARNING]
> **CRITICAL: Code results vs README claims**
> 
> The README claims Word2Vec + Random Forest achieved **97.50%** accuracy. However, the actual `model_results.csv` generated by the pipeline shows **83.92%**. The README's figures appear to be aspirational or from a different training run, NOT from the current trained models.
> 
> **In an interview, use the actual CSV results (83.92%) unless you retrain and verify the 97.50% claim.**

### 9.3 Why the Models Performed Differently — Technical Analysis

#### TF-IDF + Logistic Regression: Best at 89.00% — Why?

1. **High-dimensional linear separability**: With 10,000 TF-IDF features (including bigrams like "not good", "very bad"), the positive and negative review clusters are nearly linearly separable in this high-dimensional space
2. **Sublinear TF**: Log-scaled term frequencies prevent common words from dominating
3. **Bigrams capture negation**: "not good" as a single feature directly encodes negative sentiment
4. **L2 regularization prevents overfitting**: With 10K features and 40K samples, regularization is essential
5. **Mature, well-optimized algorithm**: LBFGS solver efficiently finds the optimal hyperplane

#### BiLSTM: 87.45% — Why second?

1. **Bidirectional context helps**: Can relate "not" at position 5 to "good" at position 50 by reading both directions
2. **Sequence awareness**: Understands word ORDER, not just word presence
3. **But limited by**: Only 5 epochs of training, 200-token truncation, relatively small embedding dimension (128), and small training set for deep learning (40K samples is on the small side for neural nets)

#### TF-IDF + XGBoost: 84.84% — Why lower than LR?

1. **200 trees with depth 6 may not be enough**: XGBoost needs careful tuning. The default hyperparameters may not be optimal for this sparse, high-dimensional input
2. **Sparse input challenge**: Decision trees make axis-aligned splits. In 10,000 sparse dimensions, finding the best split is computationally expensive
3. **XGBoost excels on dense, lower-dimensional data**: It's designed more for tabular data with tens to hundreds of features, not 10,000 sparse features

#### Word2Vec + Random Forest: 83.92% — Why lower than expected?

1. **Information loss from averaging**: Averaging word vectors for a 200-word review into ONE 100-dim vector loses a LOT of information. "This movie was good" and "Was this movie good?" produce the same vector
2. **Order completely lost**: "not good" and "good not" produce the same document vector
3. **Random Forest on 100 features**: Only 100 dimensions to work with, compared to LR's 10,000
4. **But still decent**: Word2Vec captures semantic similarity (synonyms cluster together), which helps generalization

#### LSTM: 53.30% — Why so low?

This is a **red flag** — barely above random chance (50%). Technical reasons:

1. **Only 5 epochs**: LSTMs typically need 15-30 epochs for convergence. With EarlyStopping at `patience=2`, training may have stopped at epoch 2-3
2. **High recall (95.66%) but low precision (51.89%)**: The model is predicting almost everything as Positive. This is a classic sign of **insufficient training** — the model hasn't learned to discriminate
3. **200-token truncation**: Many IMDB reviews are 500+ words. Truncation loses critical information
4. **Learning from scratch**: The embedding layer starts randomly; combined with few epochs, embeddings haven't converged
5. **Possible vanishing gradients**: Unidirectional LSTM may struggle with 200-step sequences

> **What you should remember for the interview:**
> - TF-IDF + LR won because high-dimensional sparse spaces are often linearly separable
> - LSTM performed poorly likely due to insufficient training (5 epochs is very few for sequential models)
> - The README claims 97.50% for W2V+RF but actual results show 83.92% — be honest about this discrepancy
> - High recall + low precision = model is predicting everything as one class
> - F1 score is the most fair single metric because the dataset is balanced

---

## SECTION 10: WHY WORD2VEC + RANDOM FOREST — Honest Analysis

> [!IMPORTANT]
> The README claims W2V + RF achieved 97.50%. The actual model_results.csv shows 83.92%. This section discusses both the **theoretical** strengths and the **actual** results.

### 10.1 Theoretical strengths of W2V + RF

#### Semantic representation

Word2Vec creates a **continuous vector space** where:
- "excellent", "fantastic", "amazing" → all nearby in vector space
- "terrible", "horrible", "awful" → all nearby but far from the positive cluster

This means the model generalizes across vocabulary. A review using "splendid" (rare word) gets treated similarly to one using "great" (common word) because their vectors are close.

#### Dense vectors vs sparse vectors

| Aspect | TF-IDF (Sparse) | Word2Vec (Dense) |
|---|---|---|
| Dimensions | 10,000 | 100 |
| Values per feature | Mostly 0 | All non-zero |
| Captures synonyms | No ("great" ≠ "excellent") | Yes (similar vectors) |
| Captures word order | Partially (bigrams) | No (averaged) |
| Memory efficiency | Sparse format needed | Always dense |

#### Random Forest's non-linear decision boundaries

In a 100-dimensional dense space, the positive/negative clusters may NOT be linearly separable (unlike the 10K sparse TF-IDF space). Random Forest can learn **complex, non-linear boundaries** through its tree ensemble.

### 10.2 Why the actual results differ from the README claims

**INFERRED** — Possible explanations for the discrepancy:

1. The README figures may be from a different training run with different hyperparameters or more data
2. The current model may have been trained on a subset or with different random seeds
3. The 97.50% figure might represent training accuracy (overfitting), not test accuracy
4. The README might have been written based on anticipated results or published benchmarks

### 10.3 Why LSTM/BiLSTM did worse in ACTUAL results

1. **Epoch starvation**: 5 epochs is severely insufficient for LSTMs. Published LSTM results on IMDB typically use 15-25 epochs
2. **200-token cap**: Average IMDB review is 230+ words. Truncating to 200 tokens (after preprocessing removes stopwords, actual token count is lower) loses the tail of reviews where conclusions/summaries often express the strongest sentiment
3. **No pretrained embeddings**: The model learns embeddings from scratch on 40K samples. Using GloVe or Word2Vec pretrained embeddings would dramatically improve performance
4. **Small dataset for deep learning**: 40K training samples is considered small for neural networks. LSTMs typically shine with 100K+ samples

### 10.4 Limitations — Important for interviews

You should NOT claim "Word2Vec + Random Forest is inherently better than LSTM." Because:

1. **With more epochs and pretrained embeddings**, LSTM/BiLSTM consistently achieves **87-92%** on IMDB in published research
2. **Transformer models (BERT, RoBERTa)** achieve **95%+** on IMDB, proving that sequence models CAN outperform bag-of-words approaches
3. The Word2Vec + RF result in the CSV (83.92%) is actually **lower** than TF-IDF + LR (89.00%)
4. Random Forest's 97.50% claim needs verification — it may reflect overfitting (Random Forest with `max_depth=None` can easily overfit)

> **What you should remember for the interview:**
> - Be honest: "In my experiments, TF-IDF + LR performed best at 89%. The README claims higher for W2V+RF but the actual saved results show 83.92%"
> - Explain WHY theoretically W2V+RF could work well (semantic vectors + non-linear classifier)
> - But acknowledge limitations: averaging loses order, 5 epochs is too few for LSTM
> - The key comparison insight: sparse high-dimensional + linear model vs dense low-dimensional + non-linear model
> - With proper tuning (more epochs, pretrained embeddings, longer sequences), DL models would likely outperform


"""
Feature Engineering Module.

Implements three feature extraction strategies:
A) TF-IDF Vectorization (for ML models)
B) Word2Vec Embeddings   (for Random Forest)
C) Keras Tokenizer + Padding (for LSTM / BiLSTM)
"""

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from src.utils import print_header, print_info, print_success, print_metric, print_subheader


def create_tfidf_features(X_train, X_test, max_features, save_path):
    """Create TF-IDF feature matrices from cleaned text.

    Uses unigrams + bigrams with sublinear TF scaling for
    better feature discrimination.

    Returns:
        (tfidf_train, tfidf_test, vectorizer)
    """
    print_subheader("A) TF-IDF VECTORIZATION")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents='unicode',
    )

    print_info("Fitting TF-IDF vectorizer on training data...")
    tfidf_train = vectorizer.fit_transform(X_train)
    tfidf_test = vectorizer.transform(X_test)

    print_metric("Vocabulary Size", f"{len(vectorizer.vocabulary_):,}")
    print_metric("N-gram Range", "(1, 2)")
    print_metric("Training Matrix", f"{tfidf_train.shape} (sparse)")
    print_metric("Test Matrix", f"{tfidf_test.shape} (sparse)")

    # Show sample features
    feature_names = vectorizer.get_feature_names_out()
    print_info(f"Sample features: {list(feature_names[:10])}")

    # Save
    joblib.dump(vectorizer, save_path)
    print_success(f"TF-IDF vectorizer saved to {save_path}")

    return tfidf_train, tfidf_test, vectorizer


def create_word2vec_features(X_train, X_test, vector_size, window,
                             min_count, workers, save_path):
    """Train Word2Vec embeddings and create document-level vectors.

    Each document is represented as the mean of its word vectors.

    Returns:
        (w2v_train, w2v_test, w2v_model)
    """
    print_subheader("B) WORD2VEC EMBEDDINGS")

    # Tokenize
    print_info("Tokenizing sentences...")
    train_tokens = [str(text).split() for text in X_train]
    test_tokens = [str(text).split() for text in X_test]

    # Train Word2Vec
    print_info("Training Word2Vec model (CBOW)...")
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=15,
        seed=42,
        sg=0,  # CBOW
    )

    print_metric("W2V Vocabulary Size", f"{len(w2v_model.wv):,}")
    print_metric("Embedding Dimensions", w2v_model.wv.vector_size)
    print_metric("Training Epochs", 15)

    # Create document vectors by averaging word embeddings
    def _doc_vector(tokens, model, size):
        """Average word vectors for a document. Zero-vector if no words found."""
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        if vecs:
            return np.mean(vecs, axis=0)
        return np.zeros(size)

    print_info("Creating document vectors (train)...")
    w2v_train = np.array([_doc_vector(t, w2v_model, vector_size) for t in train_tokens])

    print_info("Creating document vectors (test)...")
    w2v_test = np.array([_doc_vector(t, w2v_model, vector_size) for t in test_tokens])

    print_metric("Training Vectors Shape", w2v_train.shape)
    print_metric("Test Vectors Shape", w2v_test.shape)

    # Show similar words to 'good' as a sanity check
    if 'good' in w2v_model.wv:
        similar = w2v_model.wv.most_similar('good', topn=5)
        print_info(f"Words similar to 'good': {[w for w, _ in similar]}")

    # Save
    w2v_model.save(save_path)
    print_success(f"Word2Vec model saved to {save_path}")

    return w2v_train, w2v_test, w2v_model


def prepare_dl_sequences(X_train, X_test, max_num_words, max_seq_length, save_path):
    """Prepare tokenized and padded sequences for deep learning models.

    Returns:
        (train_padded, test_padded, tokenizer)
    """
    print_subheader("C) DEEP LEARNING SEQUENCE PREPARATION")

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print_info("Building Keras tokenizer...")
    tokenizer = Tokenizer(num_words=max_num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    print_info("Converting texts to sequences...")
    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)

    print_info("Padding sequences...")
    train_padded = pad_sequences(train_seq, maxlen=max_seq_length,
                                 padding='post', truncating='post')
    test_padded = pad_sequences(test_seq, maxlen=max_seq_length,
                                padding='post', truncating='post')

    vocab_size = min(max_num_words, len(tokenizer.word_index) + 1)

    print_metric("Full Vocabulary Size", f"{len(tokenizer.word_index):,}")
    print_metric("Effective Vocab (capped)", f"{vocab_size:,}")
    print_metric("Max Sequence Length", max_seq_length)
    print_metric("Training Sequences", train_padded.shape)
    print_metric("Test Sequences", test_padded.shape)

    # Save
    joblib.dump(tokenizer, save_path)
    print_success(f"Keras tokenizer saved to {save_path}")

    return train_padded, test_padded, tokenizer

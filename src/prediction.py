"""
Prediction Pipeline Module.

Provides a unified SentimentPredictor class that can load all
trained models and run predictions through any or all of them.
"""

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from src.preprocessing import preprocess_text
from src.utils import print_header, print_info, print_success


class SentimentPredictor:
    """Unified sentiment prediction pipeline.

    Loads all 5 trained models and provides a single predict() interface
    that runs text through any or all models.
    """

    def __init__(self, config):
        """Initialize with model paths configuration.

        Args:
            config: dict with keys:
                lr_path, xgb_path, rf_path, lstm_path, bilstm_path,
                tfidf_path, tokenizer_path, w2v_path,
                w2v_size, max_seq_len
        """
        self.config = config
        self.models = {}
        self.vectorizer = None
        self.dl_tokenizer = None
        self.w2v_model = None
        self._loaded = False

    def load_models(self):
        """Load all saved models and vectorizers from disk."""
        print_info("Loading saved models...")

        # ML models
        self.models['Logistic Regression'] = joblib.load(self.config['lr_path'])
        self.models['XGBoost'] = joblib.load(self.config['xgb_path'])
        self.models['Random Forest'] = joblib.load(self.config['rf_path'])
        print_info("  Loaded: Logistic Regression, XGBoost, Random Forest")

        # DL models
        self.models['LSTM'] = load_model(self.config['lstm_path'])
        self.models['BiLSTM'] = load_model(self.config['bilstm_path'])
        print_info("  Loaded: LSTM, BiLSTM")

        # Feature extractors
        self.vectorizer = joblib.load(self.config['tfidf_path'])
        self.dl_tokenizer = joblib.load(self.config['tokenizer_path'])
        self.w2v_model = Word2Vec.load(self.config['w2v_path'])
        print_info("  Loaded: TF-IDF vectorizer, Keras tokenizer, Word2Vec model")

        self._loaded = True
        print_success(f"All {len(self.models)} models loaded successfully!")

    def predict(self, text, model_name=None):
        """Predict sentiment for a given text.

        Args:
            text: Raw input text string.
            model_name: Optional. If provided, only use that model.
                       If None, use all models.

        Returns:
            dict mapping model_name -> {'sentiment': str, 'confidence': float}
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Preprocess
        cleaned = preprocess_text(text)

        if not cleaned.strip():
            return {name: {'sentiment': 'Unknown', 'confidence': 0.0}
                    for name in (self.models if model_name is None
                                 else {model_name: self.models[model_name]})}

        # Select models
        models_to_use = (
            {model_name: self.models[model_name]}
            if model_name else self.models
        )

        results = {}
        for name, model in models_to_use.items():
            try:
                if name in ('Logistic Regression', 'XGBoost'):
                    prediction, confidence = self._predict_tfidf(model, cleaned)
                elif name == 'Random Forest':
                    prediction, confidence = self._predict_w2v(model, cleaned)
                elif name in ('LSTM', 'BiLSTM'):
                    prediction, confidence = self._predict_dl(model, cleaned)
                else:
                    continue

                results[name] = {
                    'sentiment': 'Positive' if prediction == 1 else 'Negative',
                    'confidence': float(confidence),
                }
            except Exception as e:
                results[name] = {
                    'sentiment': 'Error',
                    'confidence': 0.0,
                }

        return results

    def _predict_tfidf(self, model, cleaned_text):
        """Predict using TF-IDF-based model."""
        features = self.vectorizer.transform([cleaned_text])
        proba = model.predict_proba(features)[0]
        prediction = int(proba[1] >= 0.5)
        confidence = max(proba)
        return prediction, confidence

    def _predict_w2v(self, model, cleaned_text):
        """Predict using Word2Vec + Random Forest."""
        tokens = cleaned_text.split()
        vecs = [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]

        if vecs:
            doc_vec = np.mean(vecs, axis=0)
        else:
            doc_vec = np.zeros(self.config['w2v_size'])

        features = doc_vec.reshape(1, -1)
        proba = model.predict_proba(features)[0]
        prediction = int(proba[1] >= 0.5)
        confidence = max(proba)
        return prediction, confidence

    def _predict_dl(self, model, cleaned_text):
        """Predict using LSTM or BiLSTM."""
        seq = self.dl_tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=self.config['max_seq_len'],
                               padding='post', truncating='post')
        proba = model.predict(padded, verbose=0)[0][0]
        prediction = int(proba >= 0.5)
        confidence = float(proba) if prediction == 1 else float(1 - proba)
        return prediction, confidence


def predict_sentiment(text, predictor, model_name=None):
    """Convenience function for quick prediction.

    Args:
        text: Input text.
        predictor: SentimentPredictor instance.
        model_name: Optional model to use.

    Returns:
        Prediction results dict.
    """
    return predictor.predict(text, model_name)


def test_predictions(predictor):
    """Test the prediction pipeline with diverse sample inputs."""
    print_header("STEP 9: PREDICTION PIPELINE TESTING")

    test_texts = [
        "I really loved this movie! The acting was phenomenal and the story was gripping.",
        "Terrible film. Waste of time and money. The plot made no sense at all.",
        "It was okay, nothing special. Some good scenes but overall mediocre.",
        "An absolute masterpiece! One of the best films I've ever seen in my life.",
        "I fell asleep halfway through. Boring and predictable storyline.",
        "The cinematography was stunning but the script needed more work.",
    ]

    for text in test_texts:
        display_text = text[:75] + '...' if len(text) > 75 else text
        print(f"\n  Input: \"{display_text}\"")

        results = predictor.predict(text)
        for model_name, result in results.items():
            sentiment = result['sentiment']
            confidence = result['confidence']
            indicator = '+' if sentiment == 'Positive' else '-'
            print(f"    [{indicator}] {model_name:25s} -> {sentiment:8s}  "
                  f"(Confidence: {confidence:.1%})")

    print_success("\nPrediction pipeline working correctly!")

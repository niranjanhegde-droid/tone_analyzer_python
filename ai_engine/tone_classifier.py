"""
tone_classifier.py
--------------------
The machine-learning half of the hybrid AI engine.

A Multinomial Naive Bayes classifier is trained (using scikit-learn)
on the bundled, hand-labeled dataset in training_data.py to predict
one of five meeting tone classes for a given sentence:

    positive | neutral | negative | aggressive | anxious

The pipeline:
    TfidfVectorizer (unigrams + bigrams, English stop-words removed)
        -> MultinomialNB

The trained model + vectorizer are cached to disk (instance/model.pkl)
using pickle so subsequent app restarts load instantly instead of
retraining. Delete the pickle file (or call train_and_save(force=True))
to retrain from scratch.
"""

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ai_engine.training_data import TRAINING_DATA

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instance", "model.pkl")

_CLASS_LABELS = ["positive", "neutral", "negative", "aggressive", "anxious"]


def _build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)),
        ("clf", MultinomialNB(alpha=0.35)),
    ])


def train_and_save(force=False):
    """Train the Naive Bayes tone classifier on the bundled dataset
    and persist it to disk. Skips retraining if a cached model already
    exists, unless force=True."""
    if os.path.exists(MODEL_PATH) and not force:
        return load_model()

    texts = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    pipeline = _build_pipeline()
    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save(force=True)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


class ToneClassifier:
    """Thin wrapper exposing predict / predict_proba for a sentence
    or a list of sentences."""

    def __init__(self):
        self.model = load_model()

    def predict(self, sentence: str) -> str:
        return self.model.predict([sentence])[0]

    def predict_proba(self, sentence: str) -> dict:
        proba = self.model.predict_proba([sentence])[0]
        classes = self.model.classes_
        return {cls: float(p) for cls, p in zip(classes, proba)}

    def predict_batch(self, sentences: list) -> list:
        if not sentences:
            return []
        preds = self.model.predict(sentences)
        probas = self.model.predict_proba(sentences)
        classes = self.model.classes_
        results = []
        for pred, proba_row in zip(preds, probas):
            results.append({
                "label": pred,
                "confidence": float(max(proba_row)),
                "distribution": {cls: float(p) for cls, p in zip(classes, proba_row)},
            })
        return results


if __name__ == "__main__":
    # quick manual sanity check when run directly: python -m ai_engine.tone_classifier
    clf = ToneClassifier()
    samples = [
        "This is completely unacceptable, fix it right now.",
        "Great job team, really proud of this release.",
        "I'm not sure this is going to work, a bit worried.",
        "The report is due Friday.",
    ]
    for s in samples:
        print(s, "->", clf.predict(s))

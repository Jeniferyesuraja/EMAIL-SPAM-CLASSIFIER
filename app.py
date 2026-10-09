import json
import os
import time
from pathlib import Path

import joblib
import nltk
import numpy as np
from flask import Flask, jsonify, render_template, request

from preprocess import find_keywords_in_text, preprocess_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = BASE_DIR / "metrics.json"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

app = Flask(__name__)

MODEL_ALIASES = {
    "best": "best",
    "nb": "nb",
    "naive-bayes": "nb",
    "logistic": "logistic",
    "lr": "logistic",
    "svm": "svm",
    "linear-svm": "svm",
}


def ensure_nltk_data():
    """Download missing NLTK packages into the project-local data folder."""
    NLTK_DATA_DIR.mkdir(exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    packages = ["stopwords", "punkt", "punkt_tab"]
    for package in packages:
        try:
            if package == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab")
            elif package == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif package == "stopwords":
                nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download(package, quiet=True, download_dir=str(NLTK_DATA_DIR))


def load_training_bundle():
    bundle_path = MODEL_DIR / "spam_classifier_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            "Model bundle not found. Please run `python train.py` before launching the app."
        )
    return joblib.load(bundle_path)


ensure_nltk_data()
MODEL_BUNDLE = load_training_bundle()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "best_model": MODEL_BUNDLE.get("best_model_name", "unknown"),
        "version": "1.0.0",
    })


@app.get("/api/metrics")
def api_metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Model metrics are not available yet."}), 404

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return jsonify(metrics)


@app.get("/api/keywords")
def api_keywords():
    keyword_payload = MODEL_BUNDLE.get("keywords", {})
    if isinstance(keyword_payload, dict) and ("spam" in keyword_payload or "ham" in keyword_payload):
        spam = keyword_payload.get("spam", [])
        ham = keyword_payload.get("ham", [])
        normalized_spam = [{"word": item["word"], "weight": float(item["weight"])} for item in spam if isinstance(item, dict) and "word" in item]
        normalized_ham = [{"word": item["word"], "weight": float(item["weight"])} for item in ham if isinstance(item, dict) and "word" in item]
        return jsonify({"spam": normalized_spam, "ham": normalized_ham})

    legacy_keywords = MODEL_BUNDLE.get("metrics", {}).get("keywords") or keyword_payload
    spam_words = legacy_keywords.get("spam_top_words", []) if isinstance(legacy_keywords, dict) else []
    ham_words = legacy_keywords.get("ham_top_words", []) if isinstance(legacy_keywords, dict) else []
    return jsonify({
        "spam": [{"word": str(word), "weight": float(index + 1)} for index, word in enumerate(spam_words)],
        "ham": [{"word": str(word), "weight": -float(index + 1)} for index, word in enumerate(ham_words)],
    })


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    raw_text = payload.get("text", "")
    requested_model = str(payload.get("model", "best") or "best").lower()

    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "Please enter a valid email or message."}), 400

    if len(raw_text.strip()) < 4:
        return jsonify({"error": "Text is too short to classify."}), 400

    selected_model = MODEL_ALIASES.get(requested_model, "best")

    try:
        start = time.perf_counter()
        vectorizer = MODEL_BUNDLE["vectorizer"]
        model_key = selected_model if selected_model != "best" else MODEL_BUNDLE.get("best_model_key", "logistic")
        estimator = MODEL_BUNDLE["all_models"].get(model_key, MODEL_BUNDLE["best_model"])
        threshold = MODEL_BUNDLE["thresholds"].get(model_key, MODEL_BUNDLE["thresholds"].get("best", 0.5))

        prepared_text = preprocess_text(raw_text)
        features = vectorizer.transform([prepared_text])

        if hasattr(estimator, "predict_proba"):
            probability = float(estimator.predict_proba(features)[0, 1])
        else:
            score = float(estimator.decision_function(features)[0])
            probability = float(1 / (1 + np.exp(-score)))

        label = "spam" if probability >= threshold else "ham"
        confidence = float(max(probability, 1.0 - probability) * 100.0)
        processing_time_ms = round((time.perf_counter() - start) * 1000.0, 2)

        keyword_payload = MODEL_BUNDLE.get("keywords", {})
        spam_keyword_entries = keyword_payload.get("spam", []) if isinstance(keyword_payload, dict) else []
        spam_word_set = {
            item.get("word") if isinstance(item, dict) else str(item)
            for item in spam_keyword_entries
        }
        top_spam_words_found = find_keywords_in_text(raw_text, spam_word_set)

        response = {
            "label": label,
            "spam_probability": round(float(probability), 4),
            "confidence": round(confidence, 2),
            "top_spam_words_found": top_spam_words_found,
            "processing_time_ms": processing_time_ms,
        }
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": f"Classification failed: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

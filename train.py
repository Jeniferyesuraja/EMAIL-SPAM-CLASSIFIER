import csv
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import TextPreprocessor, preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_CHARTS_DIR = BASE_DIR / "static" / "charts"


def ensure_directories():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    STATIC_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_dataset():
    """Download the SMS spam dataset if absent and return the dataset path."""
    ensure_directories()

    candidate_paths = [
        DATA_DIR / "SMSSpamCollection",
        DATA_DIR / "sms_spam_collection.csv",
        DATA_DIR / "spam.csv",
        DATA_DIR / "sms.tsv",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    download_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
        "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv",
    ]

    for url in download_urls:
        try:
            archive_path = DATA_DIR / ("dataset.zip" if url.endswith(".zip") else "sms.tsv")
            urlretrieve(url, archive_path)

            if url.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as archive:
                    archive.extractall(DATA_DIR)
                extracted_files = list(DATA_DIR.rglob("SMSSpamCollection"))
                if extracted_files:
                    return extracted_files[0]
            else:
                return archive_path
        except Exception:
            continue

    raise FileNotFoundError("Unable to download the SMS spam dataset from the configured sources.")


def load_dataset():
    data_path = ensure_dataset()

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path, encoding="utf-8")
        if "label" not in df.columns:
            if "v1" in df.columns and "v2" in df.columns:
                df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
            else:
                raise ValueError(f"Unsupported dataset format: {data_path}")
    else:
        rows = []
        with data_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                label = row[0].strip().lower()
                message = "\t".join(row[1:]).strip()
                if label in {"ham", "spam"}:
                    rows.append((label, message))
        df = pd.DataFrame(rows, columns=["label", "message"])

    df = df[["label", "message"]].copy()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label", "message"]).reset_index(drop=True)

    dataset_copy = DATA_DIR / "sms_spam_collection.csv"
    df.to_csv(dataset_copy, index=False)
    return df


def build_pipeline(model):
    return Pipeline([
        ("text_prep", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            strip_accents="unicode",
            lowercase=False,
        )),
        ("model", model),
    ])


def tune_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}

    for threshold in np.linspace(0.35, 0.95, 120):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = f1_score(y_true, predictions, zero_division=0)
        score = precision - 0.6 * fpr + 0.2 * recall
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
            }

    return best_threshold, best_metrics


def evaluate_model(model, X_test, y_test, threshold):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    metrics["probabilities"] = probabilities.tolist()
    metrics["threshold"] = threshold
    return metrics, probabilities


def get_feature_coefs(model, vectorizer):
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    elif hasattr(model, "estimator_") and hasattr(model.estimator_, "coef_"):
        coef = model.estimator_.coef_[0] if model.estimator_.coef_.ndim > 1 else model.estimator_.coef_
    elif hasattr(model, "feature_log_prob_"):
        coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return None

    return feature_names, coef


def extract_top_keywords(model, vectorizer, top_n=15):
    feature_data = get_feature_coefs(model, vectorizer)
    if feature_data is None:
        return {"spam": [], "ham": []}

    feature_names, coefficients = feature_data
    ordered = [
        {"word": str(feature_names[index]), "weight": float(coefficients[index])}
        for index in range(len(feature_names))
    ]

    spam_terms = [entry for entry in ordered if entry["weight"] > 0]
    spam_terms.sort(key=lambda item: item["weight"], reverse=True)

    ham_terms = [entry for entry in ordered if entry["weight"] < 0]
    ham_terms.sort(key=lambda item: abs(item["weight"]), reverse=True)

    return {
        "spam": spam_terms[:top_n],
        "ham": ham_terms[:top_n],
    }


def save_confusion_matrix(model_name, matrix, label_names=("Ham", "Spam")):
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_names), yticklabels=list(label_names))
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = STATIC_CHARTS_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path.relative_to(BASE_DIR))


def save_model_comparison_chart(metrics_by_model):
    labels = list(metrics_by_model.keys())
    f1_scores = [metrics_by_model[name]["f1"] for name in labels]
    accuracy_scores = [metrics_by_model[name]["accuracy"] for name in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, accuracy_scores, width=width, label="Accuracy", color="#7c3aed")
    ax.bar(x + width / 2, f1_scores, width=width, label="F1 Score", color="#22d3ee")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = STATIC_CHARTS_DIR / "model_comparison.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path.relative_to(BASE_DIR))


def save_class_distribution_chart(labels_series):
    counts = labels_series.value_counts().sort_index()
    labels = ["Ham", "Spam"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    plt.figure(figsize=(7, 4.5))
    sns.barplot(x=labels, y=values, palette=["#34d399", "#f87171"])
    plt.title("Class Distribution")
    plt.ylabel("Samples")
    plt.tight_layout()
    path = STATIC_CHARTS_DIR / "class_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path.relative_to(BASE_DIR))


def save_keyword_chart(spam_keywords, ham_keywords):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    spam_entries = spam_keywords[:15]
    ham_entries = ham_keywords[:15]

    spam_words = [entry["word"] for entry in spam_entries[::-1]]
    spam_weights = [entry["weight"] for entry in spam_entries[::-1]]
    ham_words = [entry["word"] for entry in ham_entries[::-1]]
    ham_weights = [entry["weight"] for entry in ham_entries[::-1]]

    axes[0].barh(spam_words, spam_weights, color="#f472b6")
    axes[0].set_title("Top Spam Keywords")
    axes[0].set_xlabel("Weight")
    axes[0].grid(axis="x", linestyle="--", alpha=0.25)

    axes[1].barh(ham_words, ham_weights, color="#38bdf8")
    axes[1].set_title("Top Ham Keywords")
    axes[1].set_xlabel("Weight")
    axes[1].grid(axis="x", linestyle="--", alpha=0.25)

    for axis in axes:
        axis.set_facecolor("#0B1220")
        axis.tick_params(colors="#e2e8f0")

    fig.patch.set_facecolor("#0B1220")
    plt.tight_layout()
    path = STATIC_CHARTS_DIR / "keyword_rankings.png"
    plt.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close()
    return str(path.relative_to(BASE_DIR))


def train_models():
    ensure_directories()
    dataset = load_dataset()
    X = dataset["message"].fillna("").astype(str)
    y = dataset["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_specs = {
        "nb": MultinomialNB(),
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42),
        "svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3,
        ),
    }

    pipeline_map = {}
    threshold_map = {}
    metrics_by_model = {}

    for name, model in model_specs.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        pipeline_map[name] = pipeline

        test_probs = pipeline.predict_proba(X_test)[:, 1]
        threshold, _ = tune_threshold(y_test, test_probs)
        threshold_map[name] = float(threshold)

        metrics, _ = evaluate_model(pipeline, X_test, y_test, threshold)
        cv_scores = cross_val_score(
            build_pipeline(model),
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1",
        )
        metrics["cross_val_f1_mean"] = float(np.mean(cv_scores))
        metrics["cross_val_f1_std"] = float(np.std(cv_scores))
        metrics["model_name"] = name

        metrics_by_model[{
            "nb": "Naive Bayes",
            "logistic": "Logistic Regression",
            "svm": "Linear SVM",
        }[name]] = metrics

        save_confusion_matrix(
            {
                "nb": "Naive Bayes",
                "logistic": "Logistic Regression",
                "svm": "Linear SVM",
            }[name],
            np.array(metrics["confusion_matrix"]),
        )

    best_model_name = max(
        metrics_by_model,
        key=lambda key: (
            metrics_by_model[key]["f1"],
            -metrics_by_model[key]["false_positive_rate"],
            metrics_by_model[key]["accuracy"],
        ),
    )
    best_key = {
        "Naive Bayes": "nb",
        "Logistic Regression": "logistic",
        "Linear SVM": "svm",
    }[best_model_name]
    best_pipeline = pipeline_map[best_key]
    best_threshold = float(threshold_map[best_key])

    keyword_summary = extract_top_keywords(
        best_pipeline.named_steps["model"],
        best_pipeline.named_steps["tfidf"],
        top_n=15,
    )

    save_model_comparison_chart(metrics_by_model)
    save_class_distribution_chart(y)
    save_keyword_chart(keyword_summary["spam"], keyword_summary["ham"])

    model_bundle = {
        "best_model_name": best_model_name,
        "best_model_key": best_key,
        "best_model": best_pipeline.named_steps["model"],
        "vectorizer": best_pipeline.named_steps["tfidf"],
        "all_models": {key: pipeline.named_steps["model"] for key, pipeline in pipeline_map.items()},
        "thresholds": {key: float(value) for key, value in threshold_map.items()},
        "threshold": best_threshold,
        "keywords": keyword_summary,
        "metrics": {
            "best_model": best_model_name,
            "threshold": best_threshold,
            "keywords": keyword_summary,
            "models": metrics_by_model,
            "charts": {
                "model_comparison": "/static/charts/model_comparison.png",
                "class_distribution": "/static/charts/class_distribution.png",
                "keyword_rankings": "/static/charts/keyword_rankings.png",
                "confusion_matrix_nb": "/static/charts/confusion_matrix_Naive_Bayes.png",
                "confusion_matrix_logistic": "/static/charts/confusion_matrix_Logistic_Regression.png",
                "confusion_matrix_svm": "/static/charts/confusion_matrix_Linear_SVM.png",
            },
        },
    }

    joblib.dump(model_bundle, MODELS_DIR / "spam_classifier_bundle.joblib")
    joblib.dump(best_pipeline.named_steps["model"], MODELS_DIR / "best_model.joblib")
    joblib.dump(best_pipeline.named_steps["tfidf"], MODELS_DIR / "vectorizer.joblib")

    metrics_payload = {
        "best_model": best_model_name,
        "threshold": best_threshold,
        "keywords": keyword_summary,
        "charts": {
            "model_comparison": "/static/charts/model_comparison.png",
            "class_distribution": "/static/charts/class_distribution.png",
            "keyword_rankings": "/static/charts/keyword_rankings.png",
            "confusion_matrix_nb": "/static/charts/confusion_matrix_Naive_Bayes.png",
            "confusion_matrix_logistic": "/static/charts/confusion_matrix_Logistic_Regression.png",
            "confusion_matrix_svm": "/static/charts/confusion_matrix_Linear_SVM.png",
        },
        "models": metrics_by_model,
    }

    with (BASE_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    print(f"Training complete. Best model: {best_model_name} (threshold={best_threshold:.4f})")
    return metrics_payload


if __name__ == "__main__":
    train_models()

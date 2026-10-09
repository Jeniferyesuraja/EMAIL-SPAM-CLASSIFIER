import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
URL_RE = re.compile(r"http\S+|www\.\S+")
NUMBER_RE = re.compile(r"\d+")
NON_ALNUM_RE = re.compile(r"[^a-zA-Z\s]")


def normalize_text(text: str) -> str:
    """Lowercase and strip noise such as URLs, numbers, and punctuation."""
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = URL_RE.sub(" ", normalized)
    normalized = NUMBER_RE.sub(" ", normalized)
    normalized = NON_ALNUM_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def preprocess_text(text: str) -> str:
    """Apply cleaning, tokenization, stopword removal, and stemming."""
    normalized = normalize_text(text)
    if not normalized:
        return ""

    tokens = word_tokenize(normalized)
    processed_tokens = []
    for token in tokens:
        if token.isalpha() and len(token) > 2 and token not in STOPWORDS:
            processed_tokens.append(STEMMER.stem(token))

    return " ".join(processed_tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """A lightweight sklearn transformer for preprocessing raw text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):
            return [preprocess_text(X)]
        return [preprocess_text(item) for item in X]


def find_keywords_in_text(text: str, keyword_set: set[str]) -> list[str]:
    """Return matching keywords from a supplied keyword set."""
    if not text:
        return []
    tokens = set(preprocess_text(text).split())
    return sorted(tokens.intersection(keyword_set))

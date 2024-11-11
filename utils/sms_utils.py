import re

import numpy as np
import pandas as pd
from models.di import LanStemmer, Lemmatizer, PortStemmer, StopWords
from imblearn.over_sampling import RandomOverSampler


def load_data_from_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df: pd.DataFrame = pd.read_csv(csv_path)
    df["content"] = df["content"].str.lower()
    df.dropna(inplace=True)

    X: np.ndarray = df.iloc[:, 1].values
    Y: np.ndarray = df.iloc[:, 0].values
    Y = np.array([1 if i == "spam" else 0 for i in Y])

    return X, Y


def balance_dataset(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    random_overSampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = random_overSampler.fit_resample([[x] for x in X], Y)
    X_resampled = [x[0] for x in X_resampled]
    return X_resampled, y_resampled


def text_preprocess(t: str, stop_words, stemmer) -> str:
    words = re.sub("[^a-zA-Z]", " ", t)
    words = [word.lower() for word in words.split() if word.lower() not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def preprocess_text(text: str, method: str = "porter") -> str:
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [
        word.lower() for word in words.split() if word.lower() not in StopWords().words
    ]

    if method == "porter":
        words = [PortStemmer().stemmer.stem(word) for word in words]
    elif method == "lancaster":
        words = [LanStemmer().stemmer.stem(word) for word in words]
    elif method == "lemmatize":
        words = [Lemmatizer().lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

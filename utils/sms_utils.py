import re
import string

import nltk
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
    text = nltk.word_tokenize(text)  # Create tokens
    text = " ".join(text)  # Join tokens
    text = [
        char for char in text if char not in string.punctuation
    ]  # Remove punctuations
    text = "".join(text)  # Join the leters
    text = [
        char for char in text if char not in re.findall(r"[0-9]", text)
    ]  # Remove Numbers
    text = "".join(text)  # Join the leters
    text = [
        word.lower() for word in text.split() if word.lower() not in StopWords().words
    ]  # Remove common english words (I, you, we,...)
    text = " ".join(text)  # Join the leters

    if method == "porter":
        text = list(map(lambda x: PortStemmer().stemmer.stem(x), text.split()))
    elif method == "lancaster":
        text = list(map(lambda x: LanStemmer().stemmer.stem(x), text.split()))
    elif method == "lemmatize":
        text = list(map(lambda x: Lemmatizer().lemmatizer.lemmatize(x), text.split()))

    return " ".join(text)  # error word


def detect_general_phone_number(text):
    phone_pattern = re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"
    )
    return phone_pattern.finditer(text)


def detect_email_addresses(text):
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    return email_pattern.finditer(text)


def detect_urls(text):
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return url_pattern.finditer(text)

import os
import pickle

import numpy as np
from models.di import TfidfVectorizerFactory
from models.sms_classifier import SMSClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class SMSMLClassifier(SMSClassifier):
    def __init__(self, model_name, model_dir):
        super().__init__(model_name, model_dir)
        self.model = None
        self.vectorizer: TfidfVectorizer = None
        self.vectorizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-vectorizer.pkl"
        )

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        # vectorize
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizerFactory().vectorizer
        self.vectorizer.fit(X.copy())
        X_tfidf_transformed = self.vectorizer.transform(X.copy())

        # train
        self.model.fit(X_tfidf_transformed, Y)

    def predict(self, X: np.ndarray) -> np.ndarray[int]:
        X_tfidf_transformed = self.vectorizer.transform(X.copy())
        return self.model.predict(X_tfidf_transformed)

    def predict_percent(self, X: np.ndarray) -> np.ndarray[float]:
        X_tfidf_transformed = self.vectorizer.transform(X.copy())
        Y_percent_pred = self.model.predict_proba(X_tfidf_transformed)[:, 1]
        return Y_percent_pred

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

import os
import pickle

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from models.di import TfidfVectorizerFactory
from models.sms_classifier import SMSClassifier
from utils.sms_utils import detect_email_addresses, detect_general_phone_number
from utils.utils import create_dirpath_if_not_exists


class SMSMLClassifier(SMSClassifier):
    def __init__(self, model_name, model_dir):
        super().__init__(model_name, model_dir)
        self.model = None
        self.vectorizer: TfidfVectorizer = None
        self.vectorizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-vectorizer.pkl"
        )

    def get_phone_and_email_features(self, X: np.ndarray) -> np.ndarray:  # noqa: F821
        phone_feature = np.array(
            [
                1 if next(detect_general_phone_number(text), None) is not None else 0
                for text in X
            ]
        )
        email_feature = np.array(
            [
                1 if next(detect_email_addresses(text), None) is not None else 0
                for text in X
            ]
        )
        return np.hstack((phone_feature.reshape(-1, 1), email_feature.reshape(-1, 1)))

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        # vectorize
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizerFactory().vectorizer
        self.vectorizer.fit(X.copy())
        X_tfidf_transformed = self.vectorizer.transform(X.copy())
        email_phone_features = self.get_phone_and_email_features(X)
        X_tfidf_transformed = sparse.hstack([email_phone_features, X_tfidf_transformed])

        # train
        self.model.fit(X_tfidf_transformed, Y)

    def predict(self, X: np.ndarray) -> np.ndarray[int]:
        X_tfidf_transformed = self.vectorizer.transform(X.copy())
        email_phone_features = self.get_phone_and_email_features(X)
        X_tfidf_transformed = sparse.hstack([email_phone_features, X_tfidf_transformed])

        return self.model.predict(X_tfidf_transformed)

    def predict_percent(self, X: np.ndarray) -> np.ndarray[float]:
        X_tfidf_transformed = self.vectorizer.transform(X.copy())
        email_phone_features = self.get_phone_and_email_features(X)
        X_tfidf_transformed = sparse.hstack([email_phone_features, X_tfidf_transformed])
        Y_percent_pred = self.model.predict_proba(X_tfidf_transformed)[:, 1]
        return Y_percent_pred

    def save(self):
        create_dirpath_if_not_exists(self.model_path)
        create_dirpath_if_not_exists(self.vectorizer_path)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

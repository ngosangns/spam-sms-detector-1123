import pickle

import numpy as np
import pandas as pd

from models.url_classifier import URLClassifier


class URLMLClassifier(URLClassifier):
    def __init__(self, model_name, model_dir):
        super().__init__(model_name, model_dir)
        self.model = None

    def train(self, X_vectorized: pd.DataFrame, Y: pd.DataFrame) -> None:
        self.model.fit(X_vectorized, Y)

    def predict(self, X_vectorized: pd.DataFrame) -> np.ndarray[int]:
        Y_pred = self.model.predict(X_vectorized)
        return np.array(Y_pred)

    def predictPercent(self, obj):  # 1.0 is 100% phishing, 0.0 is 100% ham
        # y_pro_non_phishing = self.model.predict_proba(obj)[0, 1]
        y_pro_phishing = self.model.predict_proba(obj)[0, 0]
        return y_pro_phishing

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

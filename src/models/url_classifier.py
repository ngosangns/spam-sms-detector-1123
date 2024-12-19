import os
import numpy as np


class URLClassifier:
    def __init__(self, model_name, model_dir):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = os.path.join(
            self.model_dir, f"url-{self.model_name}-model.pkl"
        )

    def train(self, X: np.ndarray, Y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray[int]:
        pass

    def predict_percent(
        self, X_vectorized
    ) -> np.ndarray[float]:  # 1.0 is 100% phishing, 0.0 is 100% ham
        pass

    def save(self):
        pass

    def load(self):
        pass

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
)

from models.di import SingletonMeta
from models.sms_ml_classifier import SMSMLClassifier


class SMSMLRandomForestClassifier(SMSMLClassifier):
    def __init__(self, model_dir):
        super().__init__("random_forest", model_dir)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def predict_percent(self, X: np.ndarray) -> np.ndarray[float]:
        X_tfidf_transformed = self.vectorizer.transform(X.copy())
        Y_percent_pred = self.model.predict_proba(X_tfidf_transformed)[:, 1]
        return Y_percent_pred


class SingletonSMSMLRandomForestClassifier(
    SMSMLRandomForestClassifier, metaclass=SingletonMeta
):
    pass

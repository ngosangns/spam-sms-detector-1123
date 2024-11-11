from sklearn.ensemble import (
    RandomForestClassifier,
)

from models.di import SingletonMeta
from models.sms_ml_classifier import SMSMLClassifier


class SMSMLRandomForestClassifier(SMSMLClassifier):
    def __init__(self, model_dir):
        super().__init__("random_forest", model_dir)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)


class SingletonSMSMLRandomForestClassifier(
    SMSMLRandomForestClassifier, metaclass=SingletonMeta
):
    pass

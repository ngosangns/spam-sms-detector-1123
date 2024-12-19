from sklearn.ensemble import (
    GradientBoostingClassifier,
)

from models.di import SingletonMeta
from models.sms_ml_classifier import SMSMLClassifier


class SMSMLGradientBoostingClassifier(SMSMLClassifier):
    def __init__(self, model_dir):
        super().__init__("gradient_boosting", model_dir)
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)


class SingletonSMSMLGradientBoostingClassifier(
    SMSMLGradientBoostingClassifier, metaclass=SingletonMeta
):
    pass

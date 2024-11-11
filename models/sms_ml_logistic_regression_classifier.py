from sklearn.linear_model import LogisticRegression

from models.di import SingletonMeta
from models.sms_ml_classifier import SMSMLClassifier


class SMSMLLogisticRegressionClassifier(SMSMLClassifier):
    def __init__(self, model_dir):
        super().__init__("logistic_regression", model_dir)
        self.model = LogisticRegression(random_state=42)


class SingletonSMSMLLogisticRegressionClassifier(
    SMSMLLogisticRegressionClassifier, metaclass=SingletonMeta
):
    pass

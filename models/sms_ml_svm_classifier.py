from sklearn.svm import SVC

from models.di import SingletonMeta
from models.sms_ml_classifier import SMSMLClassifier


class SMSMLSVMClassifier(SMSMLClassifier):
    def __init__(self, model_dir):
        super().__init__("svm", model_dir)
        self.model = SVC(
            kernel="rbf", C=10, gamma="scale", random_state=42, probability=True
        )


class SingletonSMSMLSVMClassifier(SMSMLSVMClassifier, metaclass=SingletonMeta):
    pass

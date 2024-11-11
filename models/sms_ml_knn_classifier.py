from sklearn.neighbors import KNeighborsClassifier

from models.di import SingletonMeta
from models.sms_ml_classifier import SMSMLClassifier


class SMSMLKNNClassifier(SMSMLClassifier):
    def __init__(self, model_dir):
        super().__init__("knn", model_dir)
        self.model = KNeighborsClassifier(n_neighbors=5)


class SingletonSMSMLKNNClassifier(SMSMLKNNClassifier, metaclass=SingletonMeta):
    pass

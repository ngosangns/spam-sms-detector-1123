from sklearn.model_selection import (
    GridSearchCV,
)
from sklearn.svm import SVC

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLSVMClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("svm", model_dir)

        self.model = GridSearchCV(SVC(), {"gamma": [0.1], "kernel": ["rbf", "linear"]})


class SingletonURLMLSVMClassifier(URLMLSVMClassifier, metaclass=SingletonMeta):
    pass

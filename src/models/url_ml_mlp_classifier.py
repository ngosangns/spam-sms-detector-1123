from sklearn.neural_network import MLPClassifier

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLMLPClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("mlp", model_dir)
        self.model = MLPClassifier()


class SingletonURLMLMLPClassifier(URLMLMLPClassifier, metaclass=SingletonMeta):
    pass

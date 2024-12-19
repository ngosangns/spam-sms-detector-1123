from sklearn.linear_model import LogisticRegression

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLLogisticRegressionClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("logistic-regression", model_dir)
        self.model = LogisticRegression()


class SingletonURLMLLogisticRegressionClassifier(
    URLMLLogisticRegressionClassifier, metaclass=SingletonMeta
):
    pass

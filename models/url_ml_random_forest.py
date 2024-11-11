from sklearn.ensemble import RandomForestClassifier

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLRandomForestClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("random-forest", model_dir)
        self.model = RandomForestClassifier(n_estimators=10)


class SingletonURLMLRandomForestClassifier(
    URLMLRandomForestClassifier, metaclass=SingletonMeta
):
    pass

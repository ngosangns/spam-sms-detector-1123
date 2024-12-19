from catboost import CatBoostClassifier

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLCatBoostClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("catboost", model_dir)
        self.model = CatBoostClassifier(learning_rate=0.1)


class SingletonURLMLCatBoostClassifier(
    URLMLCatBoostClassifier, metaclass=SingletonMeta
):
    pass

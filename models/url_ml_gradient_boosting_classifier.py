from sklearn.ensemble import (
    GradientBoostingClassifier,
)

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLGradientBoostingClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("gradient-boosting", model_dir)
        self.model = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)


class SingletonURLMLGradientBoostingClassifier(
    URLMLGradientBoostingClassifier, metaclass=SingletonMeta
):
    pass

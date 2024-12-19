from sklearn.tree import DecisionTreeClassifier

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLDecisionTreeClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("decision-tree", model_dir)
        self.model = DecisionTreeClassifier(max_depth=30)


class SingletonURLMLDecisionTreeClassifier(
    URLMLDecisionTreeClassifier, metaclass=SingletonMeta
):
    pass

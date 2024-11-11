from sklearn.naive_bayes import GaussianNB

from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLNaiveBayesClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("naive-bayes", model_dir)
        self.model = GaussianNB()


class SingletonURLMLNaiveBayesClassifier(
    URLMLNaiveBayesClassifier, metaclass=SingletonMeta
):
    pass

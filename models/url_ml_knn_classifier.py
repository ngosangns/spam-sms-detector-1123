from sklearn.neighbors import KNeighborsClassifier
from models.di import SingletonMeta
from models.url_ml_classifier import URLMLClassifier


class URLMLKNNClassifier(URLMLClassifier):
    def __init__(self, model_dir):
        super().__init__("knn", model_dir)
        self.model = KNeighborsClassifier(n_neighbors=1)


class SingletonURLMLKNNClassifier(URLMLKNNClassifier, metaclass=SingletonMeta):
    pass

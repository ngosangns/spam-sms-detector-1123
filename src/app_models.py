from src.models.sms_ml_svm_classifier import SingletonSMSMLSVMClassifier
from src.models.url_ml_catboost_classifier import SingletonURLMLCatBoostClassifier


class AppModels:
    def __init__(self, model_dir):
        self.sms = SingletonSMSMLSVMClassifier(model_dir)
        self.url = SingletonURLMLCatBoostClassifier(model_dir)
        self.sms.load()
        self.url.load()


def init_app_models() -> dict:
    model_dir = "./trained_models"

    return AppModels(model_dir)

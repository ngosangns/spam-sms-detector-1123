from sklearn.naive_bayes import MultinomialNB
from models.di import SingletonMeta
from models.sms_ml_classifier import SMSMLClassifier


class SMSMLNaiveBayesClassifier(SMSMLClassifier):
    def __init__(self, model_dir):
        super().__init__("naive_bayes", model_dir)
        self.model = MultinomialNB()


class SingletonSMSMLNaiveBayesClassifier(
    SMSMLNaiveBayesClassifier, metaclass=SingletonMeta
):
    pass

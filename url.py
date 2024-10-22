from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import os

class SMSClassifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_name == 'logistic_regression':
            return LogisticRegression()
        elif self.model_name == 'random_forest':
            return RandomForestClassifier()
        elif self.model_name == 'gradient_boosting':
            return GradientBoostingClassifier()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def load_sms_data(self, directory):
        sms_data = []
        labels = []
        for file_path in glob.glob(os.path.join(directory, '*.txt')):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                lines = content.split('\n')
                if len(lines) >= 2:
                    label = lines[0].lower()
                    sms_content = '\n'.join(lines[1:])
                    sms_data.append(sms_content)
                    labels.append(label)
        return sms_data, labels

    def preprocess_and_vectorize(self, sms_data):
        return self.vectorizer.fit_transform(sms_data)

    def train(self, sms_data, labels):
        sms_vectors = self.preprocess_and_vectorize(sms_data)
        self.model.fit(sms_vectors, labels)

    def predict(self, sms_data):
        sms_vectors = self.vectorizer.transform(sms_data)
        return self.model.predict(sms_vectors)
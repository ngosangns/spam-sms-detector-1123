import os
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier

class SMSClassifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        self.model = None

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
        sms_vectors = self.vectorizer.fit_transform(sms_data)
        return sms_vectors

    def balance_dataset(self, X, y):
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        return X_resampled, y_resampled

    def save_results(self, y_test, y_pred):
        results = {
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        with open(f'./ml-results/{self.model_name}_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    def load_results(self):
        with open(f'./ml-results/{self.model_name}_results.pkl', 'rb') as f:
            return pickle.load(f)

    def train_model(self, X_train, y_train):
        raise NotImplementedError

    def evaluate_model(self, X_test, y_test):
        raise NotImplementedError

class SVMClassifier(SMSClassifier):
    def __init__(self):
        super().__init__('svm')
        self.model = SVC(kernel='linear', C=1.0, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.save_results(y_test, y_pred)

class NaiveBayesClassifier(SMSClassifier):
    def __init__(self):
        super().__init__('naive_bayes')
        self.model = MultinomialNB()

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.save_results(y_test, y_pred)

class RandomForestClassifier(SMSClassifier):
    def __init__(self):
        super().__init__('random_forest')
        self.model = SklearnRandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.save_results(y_test, y_pred)

class LogisticRegressionClassifier(SMSClassifier):
    def __init__(self):
        super().__init__('logistic_regression')
        self.model = LogisticRegression(random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.save_results(y_test, y_pred)

class KNNClassifier(SMSClassifier):
    def __init__(self):
        super().__init__('knn')
        self.model = KNeighborsClassifier(n_neighbors=5)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.save_results(y_test, y_pred)

class GradientBoostingClassifier(SMSClassifier):
    def __init__(self):
        super().__init__('gradient_boosting')
        self.model = SklearnGradientBoostingClassifier(n_estimators=100, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.save_results(y_test, y_pred)

if __name__ == "__main__":
    sms_directory = './sms-data'  # Replace with the actual directory

    classifiers = [
        SVMClassifier(),
        NaiveBayesClassifier(),
        RandomForestClassifier(),
        LogisticRegressionClassifier(),
        KNNClassifier(),
        GradientBoostingClassifier(),
    ]

    for classifier in classifiers:
        sms_data, labels = classifier.load_sms_data(sms_directory)
        X_train_data, X_test_data, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)
        
        # Fit the vectorizer on the combined training and test data
        combined_data = X_train_data + X_test_data
        classifier.vectorizer.fit(combined_data)
        
        X_train = classifier.vectorizer.transform(X_train_data)
        X_test = classifier.vectorizer.transform(X_test_data)
        X_train, y_train = classifier.balance_dataset(X_train, y_train)

        classifier.train_model(X_train, y_train)
        classifier.evaluate_model(X_test, y_test)

    # Load results and generate comparison chart
    results = {}
    for classifier in classifiers:
        results[classifier.model_name] = classifier.load_results()
        # Plotting the results
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] * 100 for model in model_names]  # Convert to percentage

    plt.figure(figsize=(12, 6))  # Set default window width to 1000 pixels (10 inches)
    plt.bar(model_names, accuracies)
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison')

    # Adding accuracy labels on top of each bar
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 1, f'{accuracy:.2f}%', ha='center', va='bottom')

    plt.show()
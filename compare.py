import os
import glob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pickle

class SMSClassifier:
    def __init__(self, model_name, result_dir):
        self.model_name = model_name
        self.result_dir = result_dir
        self.model_path = os.path.join(self.result_dir, f'{self.model_name}_model.pkl')
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

    def balance_dataset(self, X_train, y_train):
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def preprocess_data(self, sms_data, labels):
        X_train_data, X_test_data, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)
        
        combined_data = X_train_data + X_test_data
        classifier.vectorizer.fit(combined_data)

        X_train, y_train = classifier.balance_dataset(classifier.vectorizer.transform(X_train_data), y_train)
        X_test = classifier.vectorizer.transform(X_test_data)

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return y_test, y_pred

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

class SVMClassifier(SMSClassifier):
    def __init__(self, result_dir):
        super().__init__('svm', result_dir)
        self.model = SVC(kernel='linear', C=1.0, random_state=42)

class NaiveBayesClassifier(SMSClassifier):
    def __init__(self, result_dir):
        super().__init__('naive_bayes', result_dir)
        self.model = MultinomialNB()

class RandomForestClassifier(SMSClassifier):
    def __init__(self, result_dir):
        super().__init__('random_forest', result_dir)
        self.model = SklearnRandomForestClassifier(n_estimators=100, random_state=42)

class LogisticRegressionClassifier(SMSClassifier):
    def __init__(self, result_dir):
        super().__init__('logistic_regression', result_dir)
        self.model = LogisticRegression(random_state=42)

class KNNClassifier(SMSClassifier):
    def __init__(self, result_dir):
        super().__init__('knn', result_dir)
        self.model = KNeighborsClassifier(n_neighbors=5)

class GradientBoostingClassifier(SMSClassifier):
    def __init__(self, result_dir):
        super().__init__('gradient_boosting', result_dir)
        self.model = SklearnGradientBoostingClassifier(n_estimators=100, random_state=42)

class BERTClassifier(SMSClassifier):
    def __init__(self, result_dir):
        super().__init__('bert-base-uncased', result_dir)
        self.model_path = os.path.join(self.result_dir, 'bert-base-uncased-model')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

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
                    labels.append(1 if label == 'spam' else 0)

        return sms_data, labels

    def balance_dataset(self, X_train_data, y_train):
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample([[x] for x in X_train_data], y_train)
        X_resampled = [x[0] for x in X_resampled]  # Flatten the resampled X back to 1D
        return X_resampled, y_resampled

    def preprocess_data(self, sms_data, labels):
        X_train_data, X_test_data, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)
        X_train_data, y_train = self.balance_dataset(X_train_data, y_train)

        train_dataset = Dataset.from_dict({'text': X_train_data, 'label': y_train})
        test_dataset = Dataset.from_dict({'text': X_test_data, 'label': y_test})

        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True)

        X_train = train_dataset.map(tokenize_function, batched=True)
        y_train = train_dataset['label']

        X_test = test_dataset.map(tokenize_function, batched=True)
        y_test = test_dataset['label']

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, X_test):
        training_args = TrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=X_train,
            eval_dataset=X_test,
        )
        trainer.train()

    def evaluate_model(self, X_test, y_test):
        test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})
        test_dataset = test_dataset.map(lambda e: self.tokenizer(e['text'], padding='max_length', truncation=True), batched=True)

        predictions = self.model.predict(test_dataset)
        return test_dataset['label'], predictions['label_ids']
    
        # report = classification_report(test_dataset['label'], predictions['label_ids'], output_dict=True)
        # self.save_results(test_dataset['label'], predictions['label_ids'], report)

    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

if __name__ == "__main__":
    sms_directory = './sms-data'
    result_dir = './ml-models'
    is_training = True
    classifiers = [
        # SVMClassifier(result_dir),
        # NaiveBayesClassifier(result_dir),
        # RandomForestClassifier(result_dir),
        # LogisticRegressionClassifier(result_dir),
        # KNNClassifier(result_dir),
        # GradientBoostingClassifier(result_dir),
        BERTClassifier(result_dir),
    ]
    model_names = list([classifier.model_name for classifier in classifiers])
    accuracies = []

    for classifier in classifiers:
        sms_data, labels = classifier.load_sms_data(sms_directory)
        X_train, y_train, X_test, y_test = classifier.preprocess_data(sms_data, labels)

        if is_training:
            if classifier.model_name == 'bert-base-uncased':
                classifier.train_model(X_train, X_test)
            else:
                classifier.train_model(X_train, y_train)
            classifier.save_model()
        else:
            classifier.load_model()
        
        y_test, y_pred = classifier.evaluate_model(X_test, y_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy * 100)

    plt.figure(figsize=(12, 6))  # Set default window width to 1000 pixels (10 inches)
    plt.bar(model_names, accuracies)
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison')

    # Adding accuracy labels on top of each bar
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 1, f'{accuracy:.2f}%', ha='center', va='bottom')

    plt.show()

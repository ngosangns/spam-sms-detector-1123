import os
import glob
import torch
import pickle
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
from transformers import TextClassificationPipeline

class SMSClassifier:
    """
    Lớp SMSClassifier dùng để phân loại tin nhắn SMS thành spam hoặc không spam.
    Attributes:
        model_name (str): Tên của mô hình.
        result_dir (str): Thư mục lưu trữ kết quả.
        model_path (str): Đường dẫn đến file lưu trữ mô hình.
        vectorizer (TfidfVectorizer): Bộ vector hóa văn bản sử dụng TF-IDF.
        model (sklearn model): Mô hình học máy dùng để phân loại.
    Methods:
        __init__(model_name, result_dir):
            Khởi tạo đối tượng SMSClassifier với tên mô hình và thư mục kết quả.
        load_sms_data(directory):
            Tải dữ liệu SMS từ thư mục chỉ định và trả về danh sách tin nhắn và nhãn tương ứng.
        balance_dataset(X_train, y_train):
            Cân bằng tập dữ liệu huấn luyện bằng cách sử dụng kỹ thuật oversampling.
        preprocess_data(sms_data, labels):
            Tiền xử lý dữ liệu SMS và nhãn, bao gồm chia tập dữ liệu và vector hóa văn bản.
        train_model(X_train, y_train):
            Huấn luyện mô hình với dữ liệu huấn luyện.
        evaluate_model(X_test, y_test):
            Đánh giá mô hình với dữ liệu kiểm tra và trả về nhãn thực tế và nhãn dự đoán.
        save_model():
            Lưu mô hình đã huấn luyện vào file.
        load_model():
            Tải mô hình từ file.
    """
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
                    labels.append(1 if label == 'spam' else 0)

        return sms_data, labels

    def balance_dataset(self, X_train, y_train):
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def preprocess_data(self, sms_data, labels):
        X_train, X_test, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)
        
        combined_data = X_train + X_test
        classifier.vectorizer.fit(combined_data)

        X_train, y_train = classifier.balance_dataset(classifier.vectorizer.transform(X_train), y_train)
        X_test = classifier.vectorizer.transform(X_test)

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
    """
    BERTClassifier là một lớp con của SMSClassifier, sử dụng mô hình BERT để phân loại tin nhắn SMS.

    Attributes:
        model_path (str): Đường dẫn lưu trữ mô hình đã huấn luyện.
        tokenizer (BertTokenizer): Bộ tokenizer của BERT.
        model (BertForSequenceClassification): Mô hình BERT dùng để phân loại.

    Methods:
        __init__(result_dir):
            Khởi tạo BERTClassifier với đường dẫn lưu trữ kết quả.
        
        balance_dataset(X_train, y_train):
            Cân bằng tập dữ liệu huấn luyện bằng cách sử dụng kỹ thuật oversampling.
        
        preprocess_data(sms_data, labels):
            Tiền xử lý dữ liệu SMS và nhãn, chia thành tập huấn luyện và tập kiểm tra.
        
        train_model(X_train, X_test):
            Huấn luyện mô hình BERT với tập dữ liệu huấn luyện và kiểm tra.
        
        evaluate_model(X_test, y_test):
            Đánh giá mô hình trên tập dữ liệu kiểm tra và trả về nhãn thực tế và nhãn dự đoán.
        
        save_model():
            Lưu trữ mô hình và tokenizer đã huấn luyện vào đường dẫn model_path.
        
        load_model():
            Tải mô hình và tokenizer từ đường dẫn model_path.
    """
    def __init__(self, result_dir):
        super().__init__('bert-base-uncased', result_dir)
        self.model_path = os.path.join(self.result_dir, 'bert-base-uncased-model')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

    def balance_dataset(self, X_train, y_train):
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample([[x] for x in X_train], y_train)
        X_resampled = [x[0] for x in X_resampled]  # Flatten the resampled X back to 1D
        return X_resampled, y_resampled

    def preprocess_data(self, sms_data, labels):
        X_train, X_test, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)
        X_train, y_train = self.balance_dataset(X_train, y_train)

        train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
        test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

        X_train = train_dataset['text']
        y_train = train_dataset['label']

        X_test = test_dataset['text']
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
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, framework='pt', device=0 if torch.cuda.is_available() else -1)
        predictions = pipeline(X_test)
        y_pred = [pred['label'] == 'LABEL_1' for pred in predictions]

        # print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        # print(classification_report(y_test, y_pred))

        return y_test, y_pred

    def save_model(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

if __name__ == "__main__":
    SMS_DIR = './sms-data'
    RESULT_DIR = './ml-models'
    IS_TRAINING = True
    
    classifiers = [
        SVMClassifier(RESULT_DIR),
        NaiveBayesClassifier(RESULT_DIR),
        RandomForestClassifier(RESULT_DIR),
        LogisticRegressionClassifier(RESULT_DIR),
        KNNClassifier(RESULT_DIR),
        GradientBoostingClassifier(RESULT_DIR),
        # BERTClassifier(RESULT_DIR),
    ]
    model_names = list([classifier.model_name for classifier in classifiers])
    accuracies = []

    for classifier in classifiers:
        sms_data, labels = classifier.load_sms_data(SMS_DIR)
        X_train, y_train, X_test, y_test = classifier.preprocess_data(sms_data, labels)

        if IS_TRAINING:
            if classifier.model_name == 'bert-base-uncased':
                classifier.train_model(X_train, X_test)
            else:
                classifier.train_model(X_train, y_train)
            classifier.save_model()
        else:
            classifier.load_model()
        
        print(f'Evaluating {classifier.model_name}...')
        y_test, y_pred = classifier.evaluate_model(X_test, y_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
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

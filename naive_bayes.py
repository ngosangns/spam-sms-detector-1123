import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

def balance_dataset(X, y):
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled

def load_sms_data(directory):
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

def preprocess_and_vectorize(sms_data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    sms_vectors = vectorizer.fit_transform(sms_data)
    return sms_vectors, vectorizer

def train_naive_bayes(X_train, y_train):
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    sms_directory = './sms-data'  # Replace with the actual directory

    sms_data, labels = load_sms_data(sms_directory)

    X_train_data, X_test_data, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)

    X_train, vectorizer = preprocess_and_vectorize(X_train_data)
    X_test = vectorizer.transform(X_test_data)
    
    X_train, y_train = balance_dataset(X_train, y_train)

    nb_model = train_naive_bayes(X_train, y_train)

    evaluate_model(nb_model, X_test, y_test)
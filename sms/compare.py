import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
from keras.src.layers.preprocessing.text_vectorization import TextVectorization
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from models import (
    SMSBERT2Classifier,
    SMSClassifier,
    SMSGradientBoostingClassifier,
    SMSKNNClassifier,
    SMSLogisticRegressionClassifier,
    SMSLSTMClassifier,
    SMSNaiveBayesClassifier,
    SMSRandomForestClassifier,
    SMSRNNClassifier,
    SMSSVMClassifier,
)
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
)
import numpy as np
import pandas as pd


def load_data_from_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df: pd.DataFrame = pd.read_csv(csv_path)
    df["content"] = df["content"].str.lower()
    df.dropna(inplace=True)

    X: np.ndarray = df.iloc[:, 1].values
    Y: np.ndarray = df.iloc[:, 0].values
    Y = np.array([1 if i == "spam" else 0 for i in Y])

    return X, Y


if __name__ == "__main__":
    CSV_PATH = "../data/sms-data.csv"
    RESULT_DIR = "../ml-models"
    IS_TRAINING = False

    # random_overSampler = RandomOverSampler(random_state=42)
    # stop_words = stopwords.words("english")

    # # stemmer
    # port_stemmer = PorterStemmer()
    # lan_stemmer = LancasterStemmer()
    # lemmatizer = WordNetLemmatizer()

    # # tokenizer
    # count_vectorizer = CountVectorizer(ngram_range=(2, 6), max_features=3000)
    # bert_tokenize_max_length = 64
    # bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    # text_vectorization = TextVectorization(
    #     output_mode="int",
    #     max_tokens=1000,
    #     output_sequence_length=1000,
    # )
    # keras_tokenizer = Tokenizer(num_words=1000)

    classifiers: list[SMSClassifier] = [
        SMSSVMClassifier(RESULT_DIR),
        SMSNaiveBayesClassifier(RESULT_DIR),
        SMSRandomForestClassifier(RESULT_DIR),
        SMSLogisticRegressionClassifier(RESULT_DIR),
        SMSKNNClassifier(RESULT_DIR),
        SMSGradientBoostingClassifier(RESULT_DIR),
        SMSLSTMClassifier(RESULT_DIR),
        SMSBERT2Classifier(RESULT_DIR),
        SMSRNNClassifier(RESULT_DIR),
    ]
    model_names = [classifier.model_name for classifier in classifiers]
    accuracies = []

    X, Y = load_data_from_csv(CSV_PATH)
    # X, Y = balance_dataset(random_overSampler, X, Y)

    X1, X2, Y1, Y2 = train_test_split(
        X.copy(), Y.copy(), test_size=0.2, random_state=42
    )

    for classifier in classifiers:
        if IS_TRAINING:
            classifier.train(X1.copy(), Y1.copy())
            classifier.save()
        else:
            classifier.load()

        print(f"Evaluating {classifier.model_name}...")
        Y2_pred = classifier.predict(X2.copy())
        accuracy = accuracy_score(Y2.copy(), Y2_pred)

        print(f"Accuracy: {accuracy}")
        accuracies.append(accuracy * 100)

    plt.figure(figsize=(12, 6))  # Set default window width to 1000 pixels (10 inches)
    plt.bar(model_names, accuracies)
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison")

    # Adding accuracy labels on top of each bar
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 1, f"{accuracy:.2f}%", ha="center", va="bottom")

    plt.show()

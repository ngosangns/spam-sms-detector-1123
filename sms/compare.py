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
    SMSBERTClassifier,
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
from utils import (
    bert2_build_trainer,
    bert_evaluate,
    bert_tokenize,
    bert_train,
    evaluate,
    keras_evaluate,
    keras_train,
    load_data,
    rnn_train,
    train,
)

if __name__ == "__main__":
    CSV_PATH = "../data/sms-data.csv"
    RESULT_DIR = "../ml-models"
    IS_TRAINING = False

    random_overSampler = RandomOverSampler(random_state=42)
    stop_words = stopwords.words("english")

    # stemmer
    port_stemmer = PorterStemmer()
    lan_stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()

    # tokenizer
    count_vectorizer = CountVectorizer(ngram_range=(2, 6), max_features=3000)
    bert_tokenize_max_length = 64
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    text_vectorization = TextVectorization(
        output_mode="int",
        max_tokens=1000,
        output_sequence_length=1000,
    )
    keras_tokenizer = Tokenizer(num_words=1000)

    classifiers: list[SMSClassifier] = [
        # SMSSVMClassifier(RESULT_DIR),
        # SMSNaiveBayesClassifier(RESULT_DIR),
        # SMSRandomForestClassifier(RESULT_DIR),
        # SMSLogisticRegressionClassifier(RESULT_DIR),
        # SMSKNNClassifier(RESULT_DIR),
        # SMSGradientBoostingClassifier(RESULT_DIR),
        # SMSLSTMClassifier(RESULT_DIR),
        # SMSBERTClassifier(RESULT_DIR),
        SMSBERT2Classifier(RESULT_DIR),
        # SMSRNNClassifier(RESULT_DIR),
    ]
    model_names = [classifier.model_name for classifier in classifiers]
    accuracies = []

    X, Y, df = load_data(CSV_PATH)
    # X, Y = balance_dataset(random_overSampler, X, Y)

    # for ML
    tfidf_vectorizer.fit(X.copy())
    X1, X2, Y1, Y2 = train_test_split(
        X.copy(), Y.copy(), test_size=0.2, random_state=42
    )
    X1_tfidf_transformed, X2_tfidf_transformed = tfidf_vectorizer.transform(
        X1.copy()
    ), tfidf_vectorizer.transform(X2.copy())

    # for LSTM
    D1 = (
        tf.data.Dataset.from_tensor_slices((X1.copy(), Y1.copy()))
        .shuffle(100)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    D2 = (
        tf.data.Dataset.from_tensor_slices((X2.copy(), Y2.copy()))
        .shuffle(100)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    text_vectorization.adapt(
        D1.map(lambda content, label: content)
    )  # Cause "Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence"

    # for BERT
    X1_input_ids, X1_attention_masks = bert_tokenize(
        X1.copy(), bert_tokenizer, bert_tokenize_max_length
    )
    X2_input_ids, X2_attention_masks = bert_tokenize(
        X2.copy(), bert_tokenizer, bert_tokenize_max_length
    )

    # for BERT2
    X1_df = Dataset.from_dict({"text": X1, "label": Y1})
    X2_df = Dataset.from_dict({"text": X2, "label": Y2})

    def tokenize_function(example):
        return bert_tokenizer(example["text"], padding="max_length", truncation=True)

    X1_bert_tokenized = X1_df.map(tokenize_function, batched=True)
    X2_bert_tokenized = X2_df.map(tokenize_function, batched=True)

    # for RNN
    keras_tokenizer.fit_on_texts(X.copy())
    X1_sequences = keras_tokenizer.texts_to_sequences(X1.copy())
    X1_sequence_max_length = max([len(sequence) for sequence in X1_sequences])
    X1_sequences_padded = pad_sequences(X1_sequences, maxlen=X1_sequence_max_length)
    X2_sequences = keras_tokenizer.texts_to_sequences(X2.copy())
    X2_sequences_padded = pad_sequences(X2_sequences, maxlen=X1_sequence_max_length)

    # for N-gram
    X1_count_vectorizer_transformed = count_vectorizer.fit_transform(X1.copy())
    X2_count_vectorizer_transformed = count_vectorizer.transform(X2.copy())

    for classifier in classifiers:
        if IS_TRAINING:
            if classifier.model_name == "lstm":
                classifier.build_lstm_model(text_vectorization)
                keras_train(classifier.model, D1, D2, epochs=5)
            elif classifier.model_name == "bert":
                classifier.build_bert_model(bert_tokenize_max_length)
                bert_train(classifier.model, X1_input_ids, X1_attention_masks, Y1)
            elif classifier.model_name == "bert2":
                bert2_trainer = bert2_build_trainer(
                    classifier.model, classifier.model_path
                )
                bert2_trainer.train()
            elif classifier.model_name == "rnn":
                classifier.build_rnn_model(X1_sequence_max_length)
                rnn_train(classifier.model, X1_sequences_padded, Y1)
            else:
                train(classifier.model, X1_tfidf_transformed, Y1)
                # train(classifier.model, X1_count_vectorizer_transformed, Y1)  # N-gram

            classifier.save()
        else:
            classifier.load()

        print(f"Evaluating {classifier.model_name}...")

        if classifier.model_name == "lstm":
            Y2_pred = keras_evaluate(classifier.model, X2)
        if classifier.model_name == "rnn":
            Y2_pred = keras_evaluate(classifier.model, X2_sequences_padded)
        elif classifier.model_name == "bert":
            Y2_pred = bert_evaluate(classifier.model, bert_tokenizer, X2)
        elif classifier.model_name == "bert2":
            Y2_pred = bert_evaluate(classifier.model, bert_tokenizer, X2.tolist())
        else:
            Y2_pred = evaluate(classifier.model, X2_tfidf_transformed)

        accuracy = accuracy_score(Y2, Y2_pred)

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

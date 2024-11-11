import os
import pickle

import keras
import numpy as np
import tensorflow as tf
from datasets import Dataset
from keras import Input, Model, Sequential
from keras.src.layers import LSTM, Activation, Bidirectional, Dense, Dropout, Embedding
from keras.src.layers.preprocessing.text_vectorization import TextVectorization
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.losses import BinaryCrossentropy
from keras.src.optimizers import Adam, RMSprop
from keras.src.utils import pad_sequences
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGradientBoostingClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier as SklearnRandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TFBertModel,
)
import torch
from transformers import (
    TextClassificationPipeline,
    TrainingArguments,
    Trainer,
)


# Base class for SMS classifiers
class SMSClassifier:
    def __init__(self, model_name, model_dir):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-model.pkl"
        )

    def train(self, X: np.ndarray, Y: np.ndarray):
        pass

    def predict(self, X_vectorized: np.ndarray) -> np.ndarray[int]:
        pass

    def save(self):
        pass

    def load(self):
        pass


class MLSMSClassifier(SMSClassifier):
    def __init__(self, model_name, model_dir):
        super().__init__(model_name, model_dir)
        self.model = None
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        self.vectorizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-vectorizer.pkl"
        )

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        # vectorize
        self.vectorizer.fit(X.copy())
        X_tfidf_transformed = self.vectorizer.transform(X.copy())

        # train
        self.model.fit(X_tfidf_transformed, Y)

    def predict(self, X: np.ndarray) -> np.ndarray[int]:
        X_tfidf_transformed = self.vectorizer.transform(X.copy())
        return self.model.predict(X_tfidf_transformed)

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)


# SVM classifier
class SMSSVMClassifier(MLSMSClassifier):
    def __init__(self, model_dir):
        super().__init__("svm", model_dir)
        self.model = SVC(kernel="linear", C=1.0, random_state=42)


# Naive Bayes classifier
class SMSNaiveBayesClassifier(MLSMSClassifier):
    def __init__(self, model_dir):
        super().__init__("naive_bayes", model_dir)
        self.model = MultinomialNB()


# Random Forest classifier
class SMSRandomForestClassifier(MLSMSClassifier):
    def __init__(self, model_dir):
        super().__init__("random_forest", model_dir)
        self.model = SklearnRandomForestClassifier(n_estimators=100, random_state=42)


# Logistic Regression classifier
class SMSLogisticRegressionClassifier(MLSMSClassifier):
    def __init__(self, model_dir):
        super().__init__("logistic_regression", model_dir)
        self.model = LogisticRegression(random_state=42)


# K-Nearest Neighbors classifier
class SMSKNNClassifier(MLSMSClassifier):
    def __init__(self, model_dir):
        super().__init__("knn", model_dir)
        self.model = KNeighborsClassifier(n_neighbors=5)


# Gradient Boosting classifier
class SMSGradientBoostingClassifier(MLSMSClassifier):
    def __init__(self, model_dir):
        super().__init__("gradient_boosting", model_dir)
        self.model = SklearnGradientBoostingClassifier(
            n_estimators=100, random_state=42
        )


# LSTM classifier
class SMSLSTMClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("lstm", model_dir)
        self.model_path = self.model_path + ".keras"
        self.model = None

    def train(self, X: np.ndarray, Y: np.ndarray):
        X1, X2, Y1, Y2 = train_test_split(
            X.copy(), Y.copy(), test_size=0.2, random_state=42
        )
        D = (
            tf.data.Dataset.from_tensor_slices((X.copy(), Y.copy()))
            .shuffle(100)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )
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

        # vectorize
        text_vectorization = TextVectorization(
            output_mode="int",
            max_tokens=1000,
            output_sequence_length=1000,
        )
        text_vectorization.adapt(
            D.map(lambda content, label: content)
        )  # Cause "Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence"

        if self.model is None:
            self.model = Sequential(
                [
                    text_vectorization,
                    Embedding(
                        len(text_vectorization.get_vocabulary()),
                        64,
                        mask_zero=True,
                    ),
                    Bidirectional(LSTM(64, return_sequences=True)),
                    Bidirectional(LSTM(32)),
                    Dense(64, activation="relu"),
                    Dropout(0.3),
                    Dense(1),
                ]
            )
            self.model.compile(
                loss=BinaryCrossentropy(from_logits=True),
                optimizer=Adam(1e-4),
                metrics=["accuracy"],
            )

        self.model.fit(
            D1,
            validation_data=D2,
            batch_size=128,
            epochs=10,
            validation_steps=30,
        )

    def predict(
        self, X_vectorized, batch_size: int = 128, verbose=1
    ) -> np.ndarray[int]:
        Y_pred = self.model.predict(
            X_vectorized, batch_size=batch_size, verbose=verbose
        )
        return np.array([1 if i > 0.5 else 0 for i in Y_pred])

    def save(self):
        self.model.save(self.model_path)

    def load(self):
        self.model = keras.models.load_model(self.model_path)


# BERT2 classifier
class SMSBERT2Classifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("bert2", model_dir)
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        self.tokenizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-tokenizer.pkl"
        )

    def train(self, X: np.ndarray, Y: np.ndarray):
        X1, X2, Y1, Y2 = train_test_split(
            X.copy(), Y.copy(), test_size=0.2, random_state=42
        )

        X1_df = Dataset.from_dict({"text": X1, "label": Y1})
        X2_df = Dataset.from_dict({"text": X2, "label": Y2})

        def tokenize_function(example):
            return self.tokenizer(
                example["text"], padding="max_length", truncation=True
            )

        X1_bert_tokenized = X1_df.map(tokenize_function, batched=True)
        X2_bert_tokenized = X2_df.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=X1_bert_tokenized,
            eval_dataset=X2_bert_tokenized,
        )

        trainer.train()

    def predict(self, X: np.ndarray) -> np.ndarray[int]:
        pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=0 if torch.cuda.is_available() else -1,
        )
        Y_pred = pipeline(X.tolist())
        Y_pred = np.array([y["label"] == "LABEL_1" for y in Y_pred])
        return Y_pred

    def save(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.tokenizer_path)

    def load(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)


# RNN classifier
class SMSRNNClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("rnn", model_dir)
        self.vectorizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-vectorizer.pkl"
        )
        self.sequence_max_length_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-sequence-max-length.pkl"
        )
        self.vectorizer = Tokenizer(num_words=1000)
        self.model: Model = None
        self.sequence_max_length = 0

    def build_rnn_model(self, max_sequence_length):
        # tạo một tensor đầu vào cho mô hình
        inputs = Input(name="inputs", shape=[max_sequence_length])
        # chuyển đổi các từ (số nguyên) thành các vector
        # có kích thước 50 mỗi từ với tối đa 1000 từ
        layer = Embedding(1000, 50)(inputs)

        layer = LSTM(64)(layer)
        layer = Dense(256, name="FC1")(layer)
        layer = Activation("relu")(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1, name="out_layer")(layer)
        layer = Activation("sigmoid")(layer)

        self.model = Model(inputs=inputs, outputs=layer)
        self.model.compile(
            loss="binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
        )

    def train(self, X: np.ndarray, Y: np.ndarray):
        # vectorize
        self.vectorizer.fit_on_texts(X.copy())
        X_sequences = self.vectorizer.texts_to_sequences(X.copy())
        self.sequence_max_length = max([len(sequence) for sequence in X_sequences])
        X_sequences_padded = pad_sequences(X_sequences, maxlen=self.sequence_max_length)

        if self.model is None:
            self.build_rnn_model(self.sequence_max_length)

        self.model.fit(
            X_sequences_padded,
            Y,
            batch_size=128,
            epochs=10,
            verbose=1,
            validation_split=0.2,
        )

    def predict(
        self, X: np.ndarray, batch_size: int = 128, verbose=1
    ) -> np.ndarray[int]:
        X_sequences = self.vectorizer.texts_to_sequences(X.copy())
        X_sequences_padded = pad_sequences(X_sequences, maxlen=self.sequence_max_length)

        Y_pred = self.model.predict(
            X_sequences_padded, batch_size=batch_size, verbose=verbose
        )
        return np.array([1 if i > 0.5 else 0 for i in Y_pred])

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self.sequence_max_length_path, "wb") as f:
            pickle.dump(self.sequence_max_length, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.sequence_max_length_path, "rb") as f:
            self.sequence_max_length = pickle.load(f)


# # Example of using BERT with DNN, but accuracy is not good, so I use BERT2 instead
# # BERT classifier
# class SMSBERTClassifier(SMSClassifier):
#     def __init__(self, model_dir):
#         super().__init__("bert", model_dir)
#         self.model: Model = None
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     def build_bert_model(self, max_length):
#         input_ids = tf.keras.Input(shape=(max_length,), dtype="int32")
#         attention_masks = tf.keras.Input(shape=(max_length,), dtype="int32")

#         bert_model = TFBertModel.from_pretrained("bert-base-uncased")
#         output = bert_model([input_ids, attention_masks])
#         output = output[1]
#         output = tf.keras.layers.Dense(32, activation="relu")(output)
#         output = tf.keras.layers.Dropout(0.2)(output)
#         output = tf.keras.layers.Dense(1, activation="sigmoid")(output)

#         model = tf.keras.models.Model(
#             inputs=[input_ids, attention_masks], outputs=output
#         )
#         model.compile(
#             tf.keras.optimizers.Adam(lr=1e-5),
#             loss="binary_crossentropy",
#             metrics=["accuracy"],
#         )

#         self.model = model

#     def tokenize(self, X, max_length=64):
#         input_ids = []
#         attention_masks = []

#         for x in X:
#             encoded = self.tokenizer.encode_plus(
#                 x,
#                 add_special_tokens=True,
#                 max_length=max_length,
#                 padding="max_length",
#                 truncation=True,
#                 return_attention_mask=True,
#             )
#             input_ids.append(encoded["input_ids"])
#             attention_masks.append(encoded["attention_mask"])

#         return np.array(input_ids), np.array(attention_masks)

#     def train(
#         self,
#         X: np.ndarray,
#         Y: np.ndarray,
#         validation_split=0.2,
#         epochs=3,
#         verbose=1,
#         batch_size=10,
#     ):
#         # vectorize
#         bert_tokenize_max_length = 64
#         X_input_ids, X_attention_masks = self.tokenize(X.copy(), 64)

#         if self.model is None:
#             self.build_bert_model(bert_tokenize_max_length)

#         self.model.fit(
#             [X_input_ids, X_attention_masks],
#             Y,
#             validation_split=validation_split,
#             epochs=epochs,
#             verbose=verbose,
#             batch_size=batch_size,
#         )

#     def predict(self, X) -> np.ndarray[int]:
#         pipeline = TextClassificationPipeline(
#             model=self.model,
#             tokenizer=self.tokenizer,
#             framework="pt",
#             device=0 if torch.cuda.is_available() else -1,
#         )
#         Y_pred = pipeline(X)
#         Y_pred = np.array([y["label"] == "LABEL_1" for y in Y_pred])
#         return Y_pred

#     def save(self):
#         with open(self.model_path, "wb") as f:
#             pickle.dump(self.model, f)

#     def load(self):
#         with open(self.model_path, "rb") as f:
#             self.model = pickle.load(f)

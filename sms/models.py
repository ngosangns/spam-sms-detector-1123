import os
import pickle

import tensorflow as tf
import keras
from keras import Sequential, Input, Model
from keras.src.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Activation
from keras.src.losses import BinaryCrossentropy
from keras.src.optimizers import Adam, RMSprop
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGradientBoostingClassifier,
)
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from transformers import (
    BertForSequenceClassification,
    TFBertModel,
)


class SMSClassifier:
    def __init__(self, model_name, model_dir):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-model.pkl"
        )
        self.model = None

    # for SMSLSTMClassifier
    def build_lstm_model(self, text_vectorization):
        pass

    # for SMSBERTClassifier
    def build_bert_model(self, max_length):
        pass

    # for SMSRNNClassifier
    def build_rnn_model(self, max_sequence_length):
        pass

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)


class SMSSVMClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("svm", model_dir)
        self.model = SVC(kernel="linear", C=1.0, random_state=42)


class SMSNaiveBayesClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("naive_bayes", model_dir)
        self.model = MultinomialNB()


class SMSRandomForestClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("random_forest", model_dir)
        self.model = SklearnRandomForestClassifier(n_estimators=100, random_state=42)


class SMSLogisticRegressionClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("logistic_regression", model_dir)
        self.model = LogisticRegression(random_state=42)


class SMSKNNClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("knn", model_dir)
        self.model = KNeighborsClassifier(n_neighbors=5)


class SMSGradientBoostingClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("gradient_boosting", model_dir)
        self.model = SklearnGradientBoostingClassifier(
            n_estimators=100, random_state=42
        )


class SMSLSTMClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("lstm", model_dir)
        self.model_path = self.model_path + ".keras"

    def build_lstm_model(self, text_vectorization):
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

    def save(self):
        self.model.save(self.model_path)

    def load(self):
        self.model = keras.models.load_model(self.model_path)


class SMSBERTClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("bert", model_dir)
        self.model: Model

    def build_bert_model(self, max_length):
        input_ids = tf.keras.Input(shape=(max_length,), dtype="int32")
        attention_masks = tf.keras.Input(shape=(max_length,), dtype="int32")

        bert_model = TFBertModel.from_pretrained("bert-base-uncased")
        output = bert_model([input_ids, attention_masks])
        output = output[1]
        output = tf.keras.layers.Dense(32, activation="relu")(output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(output)

        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks], outputs=output
        )
        model.compile(
            tf.keras.optimizers.Adam(lr=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model

    # def save(self):
    #     self.model.save_pretrained(self.model_path)
    #
    # def load(self):
    #     self.model = BertForSequenceClassification.from_pretrained(self.model_path)


class SMSBERT2Classifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("bert2", model_dir)
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

    def save(self):
        self.model.save_pretrained(self.model_path)

    def load(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)


class SMSRNNClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("rnn", model_dir)

    def build_rnn_model(self, max_sequence_length):
        inputs = Input(name="inputs", shape=[max_sequence_length])
        layer = Embedding(1000, 50)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256, name="FC1")(layer)
        layer = Activation("relu")(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1, name="out_layer")(layer)
        layer = Activation("sigmoid")(layer)

        self.model: Model = Model(inputs=inputs, outputs=layer)
        self.model.compile(
            loss="binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
        )

    # def save(self):
    #     self.model.save_pretrained(self.model_path)
    #
    # def load(self):
    #     self.model = BertForSequenceClassification.from_pretrained(self.model_path)

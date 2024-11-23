import os
import pickle

import numpy as np
from keras import Input, Model
from keras.src.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.optimizers import RMSprop
from keras.src.utils import pad_sequences

from models.di import KerasTokenizerFactory, SingletonMeta
from models.sms_ml_classifier import SMSClassifier
from utils.utils import create_dirpath_if_not_exists


class SMSRNNClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("rnn", model_dir)
        self.model: Model = None

        # vectorizer
        self.vectorizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-vectorizer.pkl"
        )
        self.vectorizer: Tokenizer = None

        # sequence max length
        self.sequence_max_length = 0
        self.sequence_max_length_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-sequence-max-length.pkl"
        )

    def train(self, X: np.ndarray, Y: np.ndarray):
        # vectorize
        if self.vectorizer is None:
            self.vectorizer = KerasTokenizerFactory().tokenizer
        self.vectorizer.fit_on_texts(X.copy())
        X_sequences = self.vectorizer.texts_to_sequences(X.copy())
        self.sequence_max_length = max([len(sequence) for sequence in X_sequences])
        X_sequences_padded = pad_sequences(X_sequences, maxlen=self.sequence_max_length)

        if self.model is None:
            # tạo một tensor đầu vào cho mô hình
            inputs = Input(name="inputs", shape=[self.sequence_max_length])
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

    def predict_percent(
        self, X: np.ndarray, batch_size: int = 128, verbose=1
    ) -> np.ndarray[float]:
        X_sequences = self.vectorizer.texts_to_sequences(X.copy())
        X_sequences_padded = pad_sequences(X_sequences, maxlen=self.sequence_max_length)

        Y_pred = self.model.predict(
            X_sequences_padded, batch_size=batch_size, verbose=verbose
        )
        return Y_pred[:, 0]

    def save(self):
        create_dirpath_if_not_exists(self.model_path)
        create_dirpath_if_not_exists(self.vectorizer_path)
        create_dirpath_if_not_exists(self.sequence_max_length_path)
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


class SingletonSMSRNNClassifier(SMSRNNClassifier, metaclass=SingletonMeta):
    pass

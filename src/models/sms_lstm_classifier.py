import os
import pickle

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from keras.src.layers.preprocessing.text_vectorization import TextVectorization
from keras.src.losses import BinaryCrossentropy
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

from models.di import SingletonMeta, TextVectorizationFactory
from models.sms_ml_classifier import SMSClassifier
from utils.utils import create_dirpath_if_not_exists


class SMSLSTMClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("lstm", model_dir)
        self.model_path = self.model_path + ".keras"
        self.model: Sequential = None

        self.vectorizer: TextVectorization = None
        self.vectorizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-vectorizer.pkl"
        )

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
        if self.vectorizer is None:
            self.vectorizer = TextVectorizationFactory().vectorization
        self.vectorizer.adapt(
            D.map(lambda content, label: content)
        )  # Cause "Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence"

        if self.model is None:
            self.model = Sequential(
                [
                    self.vectorizer,
                    Embedding(
                        len(self.vectorizer.get_vocabulary()),
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
        self, X: np.ndarray, batch_size: int = 128, verbose=1
    ) -> np.ndarray[int]:
        X_df = tf.data.Dataset.from_tensor_slices(X).batch(32)
        Y_pred = self.model.predict(X_df, batch_size=batch_size, verbose=verbose)
        return np.array([1 if i > 0.5 else 0 for i in Y_pred])

    def predict_percent(
        self, X: np.ndarray, batch_size: int = 128, verbose=1
    ) -> np.ndarray[float]:
        X_df = tf.data.Dataset.from_tensor_slices(X).batch(32)
        Y_pred = self.model.predict(X_df, batch_size=batch_size, verbose=verbose)
        return Y_pred[:, 0]

    def save(self):
        create_dirpath_if_not_exists(self.model_path)
        create_dirpath_if_not_exists(self.vectorizer_path)
        self.model.save(self.model_path)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        self.model = keras.models.load_model(self.model_path)
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)


class SingletonSMSLSTMClassifier(SMSLSTMClassifier, metaclass=SingletonMeta):
    pass

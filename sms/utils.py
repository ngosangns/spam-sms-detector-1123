import re

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling.base import BaseOverSampler
from keras import Model
from transformers import (
    TextClassificationPipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# from keras.src.callbacks import EarlyStopping


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load SMS data from a CSV file.

    Args:
        csv_path (str): The path to the CSV file containing the SMS data.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the SMS messages (X) and their corresponding labels (Y).
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    df["content"] = df["content"].str.lower()
    df.dropna()

    X: np.ndarray = df.iloc[:, 1].values
    Y: np.ndarray = df.iloc[:, 0].values
    Y = np.array([1 if i == "spam" else 0 for i in Y])

    return X, Y


def train(model, X: np.ndarray, Y: np.ndarray[int]) -> None:
    """
    Train the given model using the provided data.

    Args:
        model: The machine learning model to be trained.
        X (np.ndarray): The feature data for training.
        Y (np.ndarray[int]): The target labels for training.

    Returns:
        None
    """
    model.fit(X, Y)


def evaluate(model, X_transformed) -> np.ndarray[int]:
    """
    Evaluate the given model using the provided data.

    Args:
        model: The machine learning model to be evaluated.
        X_transformed: The transformed feature data for evaluation.

    Returns:
        np.ndarray[int]: The predicted labels.
    """
    Y_pred = model.predict(X_transformed)

    # print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    # print(classification_report(y_test, y_pred))

    return Y_pred


def text_preprocess(t: str, stop_words, stemmer) -> str:
    """
    Preprocess the given text by removing punctuation, stopwords, and applying stemming.

    Args:
        t (str): The text to be preprocessed.
        stop_words: A list of stopwords to be removed.
        stemmer: The stemmer to be used for stemming the words.

    Returns:
        str: The preprocessed text.
    """
    words = re.sub("[^a-zA-Z]", " ", t)
    words = [word.lower() for word in words.split() if word.lower() not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def balance_dataset(oversampler: BaseOverSampler, X, Y):
    X_resampled, y_resampled = oversampler.fit_resample([[x] for x in X], Y)
    X_resampled = [x[0] for x in X_resampled]  # Flatten the resampled X back to 1D
    return X_resampled, y_resampled


def balance_transformed_dataset(
    oversampler: BaseOverSampler, X_transformed, Y
) -> (np.ndarray, np.ndarray):
    """
    Balance the dataset using the provided oversampler.

    Args:
        X_transformed: The feature data
        Y (np.ndarray): The target labels.
        oversampler (BaseOverSampler): The oversampler to be used for balancing the dataset.

    Returns:
        tuple: A tuple containing the balanced feature data and target labels.
    """
    X_transformed, Y = oversampler.fit_resample(X_transformed, Y)
    return X_transformed, Y


def keras_evaluate(model, X, batch_size: int = 128, verbose=1) -> np.ndarray:
    """
    Evaluate the given Keras model using the provided data.

    Args:
        model: The Keras model to be evaluated.
        X: The feature data for evaluation.
        Y_factorized: The target labels for evaluation.
        batch_size (int): The batch size to be used for evaluation.

    Returns:
        np.ndarray: The evaluation results.
    """
    Y_pred = model.predict(X, batch_size=batch_size, verbose=verbose)
    return np.array([1 if i > 0.5 else 0 for i in Y_pred])


def keras_train(
    model,
    dataset,
    val_dataset,
    validation_steps=30,
    batch_size: int = 128,
    epochs: int = 10,
) -> None:
    """
    Train the given Keras model using the provided data.

    Args:
        model: The Keras model to be trained.
        dataset: The dataset to be used for training.
        batch_size (int): The batch size to be used for training.
        epochs (int): The number of epochs to be used for training.

    Returns:
        None
    """
    model.fit(
        dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        validation_steps=validation_steps,
    )


def bert_tokenize(X, tokenizer, max_length=64):
    input_ids = []
    attention_masks = []

    for x in X:
        encoded = tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    return np.array(input_ids), np.array(attention_masks)


def bert_train(model: Model, X_input_ids, X_attention_masks, Y):
    history = model.fit(
        [X_input_ids, X_attention_masks],
        Y,
        validation_split=0.2,
        epochs=3,
        verbose=1,
        batch_size=10,
    )


def bert_evaluate(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, X
) -> np.ndarray:
    """
    Evaluate a BERT model using the provided dataset.

    Args:
        model (PreTrainedModel): The BERT model to be evaluated.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for text preprocessing.
        X: The dataset to be evaluated.

    Returns:
        np.ndarray: The predicted labels.
    """
    pipeline = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=0 if torch.cuda.is_available() else -1,
    )
    Y_pred = pipeline(X)
    Y_pred = np.array([y["label"] == "LABEL_1" for y in Y_pred])
    return Y_pred


def rnn_train(
    model: Model,
    X_sequences_matrix,
    Y,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_split=0.2,
) -> None:
    """
    Train the given RNN model using the provided datasets.

    Args:
        model (Model): The RNN model to be trained.
        X_sequences_matrix: The feature data.
        Y: The target labels.

    Returns:
        None
    """
    model.fit(
        X_sequences_matrix,
        Y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)],
    )

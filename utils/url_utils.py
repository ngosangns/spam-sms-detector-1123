import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def load_data_from_csv(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(csv_path)
    X = data.drop(["Index", "class"], axis=1)
    Y = data["class"]
    return X, Y


def balance_dataset(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, Y_resampled = oversampler.fit_resample(X, Y)
    return X_resampled, Y_resampled
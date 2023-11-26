import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG
logger = logging.getLogger(__name__)


def upload_data(filename: str = "housing.csv") -> pd.DataFrame:
    return pd.read_csv(filename, delim_whitespace=True, header=None)


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """Scale all data to values from 0 to 1."""

    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)
    return pd.DataFrame(scaled_df)


def fetch_data() -> pd.DataFrame:
    column_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]
    scaled_df = scale_data(upload_data())
    scaled_df.columns = column_names
    logger.debug(f"\n{scaled_df.head()}")
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_df.drop(columns=["MEDV"]),
        scaled_df["MEDV"],
        test_size=0.3,
        random_state=42,
    )

    return (
        np.array(X_train).T,
        np.array(X_test).T,
        np.array(y_train)[np.newaxis, :],
        np.array(y_test)[np.newaxis, :],
    )


class Dataset:
    def __init__(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = fetch_data()

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from sklearn.preprocessing import StandardScaler, scale
from technical_analysis import get_ta_indicators

from typing import *


class DataSource:

    def __init__(self,
                 symbol: str = "AAPL",
                 start_date: str = "2014-01-01",
                 end_date: str = "2018-12-31",
                 train_ratio: float = 0.8,
                 n_days: int = 14,
                 save: bool = True,
                 data_path: str = None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.train_ratio = train_ratio

        self.data = self.load_data(
            data_path) if data_path else self.download_data(save)
        self.train_size = int(np.ceil(len(self.data) * train_ratio))

        self.features = self.preprocessing_data(n_days)

        self.train_data, self.test_data = self.split_train_test(
            self.data, train_ratio)
        self.train_features, self.test_features = self.split_train_test(
            self.features, train_ratio)

    def download_data(self, save: bool = True) -> pd.DataFrame:
        data = yf.download(tickers=self.symbol,
                           start=self.start_date,
                           end=self.end_date).dropna().sort_index()
        data.rename(columns=str.lower, inplace=True)

        if save:
            save_dir = Path("./data")
            save_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(save_dir / f"{self.symbol}.csv")

        return data

    def load_data(self, path: str = "") -> pd.DataFrame:
        data = pd.read_csv(path, index_col='Date').dropna().sort_index()
        data.rename(columns=str.lower, inplace=True)

        return data

    def preprocessing_data(self, data, period: int = 14) -> pd.Dataframe:
        df = get_ta_indicators(data, period)
        df = df.replace((np.inf, -np.inf), np.nan).drop(
            ['close', 'high', 'low', 'volume', 'open', 'adj close'],
            axis=1).dropna()

        # Normalized data
        sc = StandardScaler()
        scaled_df = sc.fit_transform(df)
        data = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)
        data["n_days_returns"] = df.loc[:, "n_days_returns"]
        return data

    def split_train_test(self, data, train_ratio):
        end = int(np.ceil(len(data) * train_ratio))
        return data.iloc[:end], data.iloc[end:]

    def get_state(self, t: int) -> pd.Series:
        if t > self.train_size - 1:
            raise Exception(
                "Index is out of range. Exceeds the train_mode size.")
        else:
            return self.train_features.iloc[t]

    def get_price(self, t: int) -> float:
        if t > self.train_size - 1:
            raise Exception(
                "Index is out of range. Exceeds the train_mode size.")
        else:
            return self.train_data.iloc[t]

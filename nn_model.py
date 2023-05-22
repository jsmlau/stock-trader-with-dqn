import random
import numpy as np
from pathlib import Path
from datetime import datetime
from keras.models import Sequential, save_model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense

from typing import *


class NNModel:

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_units: tuple = (256, 256),
                 learning_rate: float = 1e-4,
                 train: bool = True,
                 model_path: str = "./models/model"):
        self.state_size = state_size
        self.action_size = action_size

        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        self.model = self.build_model() if train else self.load(model_path)

    def build_model(self) -> NoReturn:
        model = Sequential(Input((self.state_size,)),
                           Dense(units=self.hidden_units[0], activation="relu"),
                           Dense(units=self.hidden_units[1], activation="relu"),
                           Dense(units=self.action_size, activation="linear"))
        model.complie(loss="mse", optimizer=Adam(lr=self.learning_rate))
        print(model.summary())

        return model

    def load(self, path):
        model = load_model(path)
        return model

    def save(self, path):
        save_model(path)

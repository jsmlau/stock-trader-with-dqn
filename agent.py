import random
import numpy as np
from collections import deque
import tensorflow as tf
from nn_model import NNModel


class DQNAgent:

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_units: tuple = (256, 256),
                 learning_rate: float = 1e-4,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_exponential_decay: float = 0.995,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 memory_size: int = 10000,
                 train_mode: bool = True,
                 model_path: str = ""):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.inventory = {"trade": None, "price": None}
        self.train_mode = train_mode

        # Initialize NN hyper-parameters
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount rate
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_exponential_decay = epsilon_exponential_decay

        # Initialize Networks
        self.model = NNModel(self.state_size, self.action_size,
                             self.hidden_units, self.learning_rate, train_mode,
                             model_path)

    def select_action(self, state):
        # uniform random policy
        if self.train_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # greedy policy
        q_values = self.model.model.predict(state)
        return np.argmax(q_values[0], axis=1)

    def replay_experience(self):
        size = len(self.memory)

        if self.batch_size > size:
            return

        mini_batch = []

        for i in range(size - self.batch_size + 1, size):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                next_q_values = np.amax(self.model.model.predict(next_state)[0])
                target = reward + self.gamma * next_q_values

        # Q(s, a)
            target_final = self.model.model.predict(state)
            target_final[0][action] = target
            self.model.model.fit(state, target_final, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_exponential_decay

    def reset_inventory(self) -> None:
        self.inventory = {'trade': None, 'price': None}

    def update_inventory(self, trade: str = None, price: float = None) -> None:
        self.inventory['trade'] = trade
        self.inventory['price'] = price

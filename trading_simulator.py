from agent import DQNAgent
from data_source import DataSource


class TradingSimulator:

    def __init__(self,
                 ticker: str = "AAPL",
                 start_date: str = "2014-01-01",
                 end_date: str = "2018-12-31",
                 train_ratio: float = 0.7,
                 period: int = 14,
                 save_data: bool = True,
                 episodes: int = 252,
                 hidden_units: tuple = (256, 256),
                 learning_rate: float = 1e-4,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_exponential_decay: float = 0.995,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 memory_size: int = 10000,
                 train_mode: bool = True,
                 model_path: str = "./models/model",
                 verbose: bool = False):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.train_ratio = train_ratio
        self.episodes = episodes
        self.model_path = model_path
        self.train_mode = train_mode
        self.verbose = verbose
        self.data_source = DataSource(self.ticker, self.start_date,
                                      self.end_date, train_ratio, period,
                                      save_data)
        state_size = self.data_source.train_features.shape[1]
        action_size = 3
        if train_mode:
            self.agent = DQNAgent(state_size, action_size, hidden_units,
                                  learning_rate, epsilon, epsilon_min,
                                  epsilon_exponential_decay, gamma, batch_size,
                                  memory_size, True, model_path)
        else:
            self.agent = DQNAgent(state_size, action_size, hidden_units,
                                  learning_rate, epsilon, epsilon_min,
                                  epsilon_exponential_decay, gamma, batch_size,
                                  memory_size, False, model_path)

        self.total_profit = 0

    def run_episode(self):
        data_size = self.data_source.train_size - 1

        for t in range(data_size):

            current_state = self.data_source.get_state(t)
            action = self.agent.select_action(current_state)

            next_state = self.data_source.get_state(t + 1)
            reward = 0

            last_trade = self.agent.inventory['trade']
            last_price = self.agent.inventory['price']
            current_price = self.data_source.get_price(t)

            # BUY
            if action == 1:
                self.agent.inventory.update_inventory(last_trade, last_price)

                if self.verbose:
                    print(f"Buy 1 share at ${current_price}")
            elif action == 2 and self.agent.inventory['trade'] == 'short':
                bought_price = self.agent.inventory['price']
                reward = last_price - bought_price
                self.total_profit += reward
                if self.verbose:
                    print(
                        f"Sell | Profit: ${reward} | Total Profit: ${self.total_profit}"
                    )

            done = True if t == data_size - 1 else False
            self.agent.memory.append(
                (current_state, action, reward, next_state, done))

            if done:
                print(f"Total Profit: {self.total_profit}")

            if self.agent.train_mode:
                self.agent.replay_experience()

    def train(self):
        # Train the agent with N episodes
        for e in range(self.episodes + 1):
            print(f"Episode: {e} / {self.episodes}")
            self.total_profit = 0
            self.agent.reset_inventory()
            self.run_episode()

            if e % 10 == 0:
                self.agent.model.save(self.model_path)

    def evaluation(self):
        self.run_episode()

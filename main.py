from trading_simulator import TradingSimulator


def main(name):

    ticker = "AAPL",
    start_date = "2018-01-01"
    end_date = "2020-12-31"
    train_ratio = 0.7
    period = 14
    save_data = True
    episodes = 252
    hidden_units = (256, 256)
    learning_rate = 1e-4
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_exponential_decay = 0.995
    gamma = 0.99
    batch_size = 32
    memory_size = 10000
    verbose = False

    model_path = "./models/model"

    simulator = TradingSimulator(ticker, start_date, end_date, train_ratio,
                                 period, save_data, episodes, hidden_units,
                                 learning_rate, epsilon, epsilon_min,
                                 epsilon_exponential_decay, gamma, batch_size,
                                 memory_size, model_path, verbose)

    # Train
    simulator.train()


if __name__ == '__main__':
    main()

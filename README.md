# Stock Trader with DQN
This project implements the Deep Q-Network (DQN) algorithm for stock trading. The DQN model utilizes n-day windows 
of closing prices and various technical indicators to make decisions on whether to buy, sell, or hold stocks at a given time. The model aims to identify profitable opportunities in the stock market.

## Representation

1. Data Source: The trader obtains stock data from the yfinance library, which provides convenient access to historical stock price data from Yahoo Finance.
2. State Space: The input of the model. The model analyzes n-days returns and extracts relevant features that capture important market trends.
3. Action Space: The output of the model. Based on the current state representation, the DQN model selects the most appropriate 
   action to take: 
   Buy, Sell, and Hold. The model uses epsilon greedy policy to balance exploration and exploitation.
4. Reward: The reward provides feedback to the model based on its trading decisions. 

# Final-Project---IronHack
 IronHACK July24 DA FT - Sebastião Ngombo

Cryptocurrency Price Prediction and Trading Strategy Optimization
Introduction
This project focuses on developing a predictive trading strategy for Bitcoin, major altcoins, and memecoins using machine learning models. By leveraging historical cryptocurrency data and technical indicators, the objective is to predict future price movements and optimize an automated trading strategy. The project employs models such as Linear Regression, Random Forest, LSTM, and XGBoost to identify the most effective predictive approach. Additionally, risk management mechanisms like stop-loss and take-profit strategies are integrated to improve the profitability and consistency of the trading strategy.

Data
Overview
The data used in this project consists of historical price data for the following cryptocurrencies:

Bitcoin (BTC/USDT)
Ethereum (ETH/USDT)
Cardano (ADA/USDT)
Dogecoin (DOGE/USDT)
Ripple (XRP/USDT)
Litecoin (LTC/USDT)
Chainlink (LINK/USDT)
Stellar (XLM/USDT)
Data Sources:
The data was collected from Binance API for different timeframes, including:

1 minute, 5 minutes, 15 minutes, 30 minutes, 1 hour, 4 hours, 8 hours, and 1 day intervals.
Initial Collection: The dataset initially covered the period from 2018 to 31st August 2024, leading to a massive dataset that significantly increased computational costs.

Final Approach: Due to time constraints, the dataset was reduced to the period from 2022 to 31st August 2024, with a 4-hour interval. This produced a dataset of 5831 rows and 20 columns.

Main Challenges:
Large Data Size: The initial dataset was too large to process efficiently, leading to long model training times.
Non-Stationarity: Time-series data for cryptocurrencies is highly volatile and non-stationary, requiring transformations such as differencing.
Missing Data: Handling missing values across various coins while maintaining the integrity of the time series.
Strengths:
High Granularity: The availability of data at multiple time intervals allows for detailed trend analysis and prediction.
Broad Asset Coverage: By including both major altcoins and memecoins, the project captures a wide spectrum of market dynamics.
Weaknesses:
Highly Volatile Data: Cryptocurrencies are inherently volatile, which adds noise and makes modeling more difficult.
Correlation: High correlation between some altcoins and BTC can reduce the diversification benefit.
Questions Addressed
How accurately can we predict cryptocurrency prices?

Using models like Linear Regression, Random Forest, and LSTM, we predicted price trends for BTC and other coins.
Conclusion: Linear Regression performed best for BTC price prediction, but performance varied across coins, with non-linear models like LSTM performing better for highly volatile assets.
Can predicted BTC prices improve the prediction of other altcoins?

We tested whether including BTC predicted prices as a feature in predicting other altcoins would enhance accuracy.
Conclusion: Including BTC predictions improved accuracy for some coins, especially those moderately correlated with BTC (e.g., XRP, LTC).
What is the most effective trading strategy?

A trading strategy was developed using predicted prices to generate buy/sell signals, with stop-loss and take-profit mechanisms added to manage risk.
Conclusion: Coins with weaker correlation to BTC and lower volatility tended to perform better in the trading strategy, reducing overall risk.
What is the best way to manage risk in a volatile market?

We implemented stop-loss and take-profit mechanisms to limit drawdowns and secure profits.
Conclusion: Risk management features helped stabilize returns, especially during periods of high volatility.
Methodology
Data Collection: Historical cryptocurrency data was collected from the Binance API for multiple timeframes. Due to the size of the dataset, we opted for a 4-hour interval from 2022 to 2024 to balance data granularity and processing time.

Data Preprocessing:

Missing Values: Missing values were handled using forward filling and backward filling to maintain the continuity of time series.
Feature Scaling: Features were scaled using MinMaxScaler to ensure the models received normalized input.
Stationarity Checks: The Dickey-Fuller test was applied to check for stationarity, and transformations (e.g., differencing) were applied to stabilize non-stationary time series.
Exploratory Data Analysis (EDA):

A correlation matrix was generated to understand the relationship between BTC and other altcoins.
Technical indicators such as SMA, RSI, and volatility were calculated and analyzed to generate features for the model.
Modeling:

Models applied: Linear Regression, Random Forest, LSTM, and XGBoost.
Hyperparameter tuning was performed using grid search for Random Forest and LSTM to optimize model performance.
Predictions were generated for all coins, with BTC prices used as a feature for predicting altcoins.
Backtesting and Trading Strategy:

A trading strategy was developed based on predicted prices, generating buy/sell signals.
KPIs calculated: Number of trades, ROI, and final portfolio balance for each coin.
Conclusions
Price Prediction: Linear Regression performed best for BTC predictions, while non-linear models like Random Forest and LSTM performed better for altcoins with high volatility.
Diversification: Including coins with weak correlation to BTC (e.g., ADA, XRP) helped improve portfolio performance by reducing risk.
Trading Strategy: The trading strategy was effective when combined with risk management features like stop-loss and take-profit mechanisms.
Risk Management: Risk management significantly improved the stability of returns, especially in highly volatile markets like cryptocurrencies.
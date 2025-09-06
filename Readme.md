#  Recommending Stocks with a High Rise Using Google Bert (Sentiment Analysis Transfer Model)

Using the Bert model used for movie review sentiment analysis (good or bad) provided in the Temsorflow example, we analyze recent Google news articles about KOSPI 200 or KOSDAQ stocks and, based on the results, recommend stocks expected to surge in the next week.

1. Find stocks that have surged (over 10%) or plummeted (under -10%) in the past week using Kiwoom's conditional search.

- Kiwoom Securities screen: Banho 0150 (conditional search) > Price Analysis > Price Condition > Stock Price Fluctuation Rate

2. Collect news articles about the stocks listed above from the past week to create training data (rapid rise: 1, sharp fall: 0). -- F:\rapid rise_stock_bert\GoogleNewsCrawling.py

3. Using the prepared training data, we run the train and test cycles to complete the model for recommending stocks with a high rise. -- F:\Rapid_Stocks_bert\train_bert.py

4. Using the model in 3, find stocks expected to surge over the next week.
- F:\Rapid_Stocks_bert\Rapid_Stock_Recommendation2.py

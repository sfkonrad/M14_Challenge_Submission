# [Module 14 Challenge Submission](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/M14_Challenge_KonradK_machine_learning_trading_bot.ipynb)

##### Konrad Kozicki
### UCB-VIRT-FIN-PT-12-2020-U-B-TTH
---

# Machine Learning Trading Bot

For this Assignment, we've assumed roles as financial advisors at one of the top five financial advisory firms in the world. Our firm constantly competes with the other major firms to manage and automatically trade assets in a highly dynamic environment. In recent years, the firm has heavily profited by using computer algorithms that can buy and sell faster than human traders.

The speed of these transactions has given our firm a competitive advantage early on. But, humans still need to specifically program these systems, which limits the systems' ability to adapt to new data. Therefore, our team is planning to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, we have enhanced the existing trading signals with machine learning algorithms that can adapt to new data.


---
---
---
---

## Instructions:

We were provided a starter code file to complete the steps that their instructions outline. The steps for this Challenge are divided as follows:

* [Establish a Baseline Performance](https://github.com/sfkonrad/M14_Challenge_Submission#establish-a-baseline-performance)

* [Tune the Baseline Trading Algorithm](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/README.md#tune-the-baseline-trading-algorithm)

* [Evaluate a New Machine Learning Classifier](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/README.md#evaluate-a-new-machine-learning-classifier)

* [Create an Evaluation Report](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/README.md#create-an-evaluation-report)

#### Establish a Baseline Performance

In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.

Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four. 

1. Import the OHLCV dataset into a Pandas DataFrame.

2. Generate trading signals using short- and long-window SMA values. 

3. Split the data into training and testing datasets.

4. Use the `SVC` classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.

5. Review the classification report associated with the `SVC` model predictions. 

6. Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.

7. Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.

8. Write your conclusions about the performance of the baseline trading algorithm in the `README.md` file that’s associated with your GitHub repository. Support your findings by using the PNG image that you saved in the previous step.

#### Tune the Baseline Trading Algorithm

In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. (You’ll choose the best by comparing the cumulative products of the strategy returns.) To do so, complete the following steps:

1. Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing the training window?

> **Hint** To adjust the size of the training dataset, you can use a different `DateOffset` value&mdash;for example, six months. Be aware that changing the size of the training dataset also affects the size of the testing dataset.

2. Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

3. Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your `README.md` file.

#### Evaluate a New Machine Learning Classifier

In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model. To do so, complete the following steps:

1. Import a new classifier, such as `AdaBoost`, `DecisionTreeClassifier`, or `LogisticRegression`. (For the full list of classifiers, refer to the [Supervised learning page](https://scikit-learn.org/stable/supervised_learning.html) in the scikit-learn documentation.)

2. Using the original training data as the baseline model, fit another model with the new classifier.

3. Backtest the new model to evaluate its performance. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your `README.md` file. Answer the following questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?

#### Create an Evaluation Report

In the previous sections, you updated your `README.md` file with your conclusions. To accomplish this section, you need to add a summary evaluation report at the end of the `README.md` file. For this report, express your final conclusions and analysis. Support your findings by using the PNG images that you created.









## Evaluate the Metrics
Based on the metrics that the portfolio evaluation DataFrame contains, we can estimate the following:

- Annualized return: A portfolio that this trading algorithm manages should yield an annualized return of about 6.67%. That is, the dollar value of the portfolio should increase by about 6.67% each year.

- Cumulative returns: The dollar value of the portfolio increased by approximately 33% over the backtesting period. We can reasonably expect a similar growth trajectory for the future.

- Annual volatility: The annual volatility of the portfolio should have a spread of about 13.78% surrounding the annualized return. This means that the portfolio might return as much as 20.45% (6.67% + 13.78%) or lose as much as −7.11% (6.67% − 13.78%) per year.

- Sharpe ratio: The Sharpe ratio, which evaluates the performance of a portfolio on a risk-adjusted basis, is 0.484. In general, a higher Sharpe ratio indicates a better risk/reward profile. We commonly get a Sharpe ratio of about 1.00 for a portfolio with a favorable risk-adjusted profile. However, a single Sharpe ratio doesn’t offer much insight. It’s best to compare it with the Sharpe ratios of other portfolios to determine which one offers the best profile.

- Sortino ratio: The Sortino ratio of our portfolio suggests a rate of about 0.685956 for the risk-adjusted annual profitability compared to the annual downside risk. As with the Sharpe ratio, a higher Sortino ratio is better. And, it’s best to compare it with the Sortino ratios of other portfolios.

Alone, these annualized return and cumulative return values indicate overall profitability. However, we need to consider the annual volatility value and the Sharpe and Sortino ratios that compare the risk to the reward. From these metrics, the overall profitability of this portfolio doesn’t seem to outweigh its risks. For a risk-averse investor, we might need to adjust the assets that we include in the portfolio to generate a similar return profile, but with the Sharpe and Sortino ratios closer to 1.00. For an investor who is risk loving and has a longer-term time horizon, the potential of a 20% return might outweigh the downside risk of the portfolio.

Now, let's zoom in even further to discover the risk/reward characteristics that we can determine from the behavior of our trading algorithm on a per-trade basis.

## Set Up the Trade-Level Risk/Reward Evaluation Metrics
Just as we created a new DataFrame to hold our portfolio-level evaluation metrics, we now create a new DataFrame, named `trade_evaluation_df`, to hold our per-trade evaluation metrics. These include the following columns:

    > - “Stock”: The name of the asset that we’re trading.

    > - “Entry Date”: The date that we entered (bought) the trade.

    > - “Exit Date”: The date that we exited (sold) the trade.

    > - “Shares”: The number of shares that we executed for the trade.

    > - “Entry Share Price”: The price of the asset when we entered the trade.

    > - “Exit Share Price”: The price of the asset when we exited the trade.

    > - “Entry Portfolio Holding”: The cost of the trade on entry (which is the number of shares multiplied by the entry share price).

    > - “Exit Portfolio Holding”: The proceeds that we made from the trade on exit (which is the number of shares multiplied by the exit share price).

    > - “Profit/Loss”: The profit or loss from the trade (which is the proceeds from the trade minus the cost of the trade).

Notice that our trading algorithm made six trades. Three were profitable, and three were unprofitable. This gives us a trade accuracy, or percentage of profitable trades to total trades, of 50%. Also, notice that the maximum loss on any trade was −$9,115. The maximum gain was $32,130. That’s pretty good! Conversely, the minimum loss was −$3,810, and the minimum gain was $2,890.
    > 

ON THE JOB
We often consider a trading algorithm to be good if it consistently maximizes its gains while minimizing its losses. A Sharpe ratio that’s greater than 1 and a trading accuracy that’s greater than 80% are popular baselines.

Overall, it seems that our DMAC trading algorithm produces a profit. However, when we dive more deeply into the risk/reward characteristics of the algorithm, the portfolio and per-trade metrics show the potential for inconsistent performance.

Evaluate the Metrics
Both the portfolio-level and the trade-level evaluation metrics reveal the risk/reward characteristics of this portfolio and algorithmic strategy. The portfolio-level evaluation metrics give us an annual volatility of 13.7% and a Sharpe ratio of 0.484. The per-trade metrics show that only one trade turned the overall profit of the portfolio from negative to positive.

With real money, we might hesitate to use this trading algorithm—unless we have quite a high risk tolerance.

Next, you’ll get to do this on your own! You'll evaluate the risk/reward characteristics of your short-position strategy in the following activity.
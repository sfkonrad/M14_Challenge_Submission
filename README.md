# [Module 14 Challenge Submission](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/M14_Challenge_KonradK_machine_learning_trading_bot.ipynb)

##### Konrad Kozicki
### UCB-VIRT-FIN-PT-12-2020-U-B-TTH
---

# Machine Learning Trading Bot

For this Assignment, we've assumed roles as financial advisors at one of the top five financial advisory firms in the world. Our firm constantly competes with the other major firms to manage and automatically trade assets in a highly dynamic environment. In recent years, the firm has heavily profited by using computer algorithms that can buy and sell faster than human traders.

The speed of these transactions has given our firm a competitive advantage early on. But, humans still need to specifically program these systems, which limits the systems' ability to adapt to new data. Therefore, our team is planning to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, we have enhanced the existing trading signals with machine learning algorithms that can adapt to new data.

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_vs_Actual_Returns.png?raw=true)

---
---
---
---

## Instructions:

We were provided a starter code file to complete the steps that their instructions outline. The steps for this Challenge are divided as follows:

* [Establish a Baseline Performance](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/README.md#establish-a-baseline-performance)

* [Tune the Baseline Trading Algorithm](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/README.md#tune-the-baseline-trading-algorithm)

* [Evaluate a New Machine Learning Classifier](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/README.md#evaluate-a-new-machine-learning-classifier)

* [Create an Evaluation Report](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/README.md#evaluation-report)

# Establish a Baseline Performance

In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.

Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four. 

1. Import the OHLCV dataset into a Pandas DataFrame.

2. Generate trading signals using short- and long-window SMA values. 

3. Split the data into training and testing datasets.

4. Use the `SVC` classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.

5. Review the classification report associated with the `SVC` model predictions. 

6. Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.

7. Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_SVM1_vs_Actual_Returns.png?raw=true)


8. Write your conclusions about the performance of the baseline trading algorithm in the `README.md` file that’s associated with your GitHub repository. Support your findings by using the PNG image that you saved in the previous step.

# Tune the Baseline Trading Algorithm

In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. (You’ll choose the best by comparing the cumulative products of the strategy returns.) To do so, complete the following steps:


1. Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing the training window?

> **Hint** To adjust the size of the training dataset, you can use a different `DateOffset` value&mdash;for example, six months. Be aware that changing the size of the training dataset also affects the size of the testing dataset.

2. Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

A Bad Tune 🎶
> ![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_SVM_Bad_Tune_Returns.png?raw=true)

3. Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your `README.md` file.

A Better One
> ![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_2SVM_vs_Actual_Returns.png?raw=true)

# Evaluate a New Machine Learning Classifier

In this section, we used the original parameters that the starter code provided. However, we applied them to the performance of a second machine learning model. To do so, we completed the following steps:

1. Import a new classifier, such as `AdaBoost`, `DecisionTreeClassifier`, or `LogisticRegression`. (For the full list of classifiers, refer to the [Supervised learning page](https://scikit-learn.org/stable/supervised_learning.html) in the scikit-learn documentation.)

2. Using the original training data as the baseline model, fit another model with the new classifier.

3. Backtest the new model to evaluate its performance. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your `README.md` file. Answer the following questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_LR_vs_Actual_Returns.png?raw=true)



# Evaluation Report

In the previous sections, we updated your `README.md` file with your conclusions. To accomplish this section, we wee required to add a summary evaluation report at the end of the `README.md` file. For this report, express your final conclusions and analysis. Support your findings by using the PNG images that you created.

### Step 3: Backtest the models to evaluate their performance. 

Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your `README.md` file. 

Answer the following questions: 
Did this new model perform better or worse than the provided baseline model? 

### ANSWER 
# THE TUNED SVM MODEL PERFORMED BETTER THAN ACTUAL AND BASELINE RETURNS.
> ![image.png](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_2SVM_vs_Actual_Returns.png?raw=true)
> 
> "Tuned SVM Returns" `1.62`:1 RETURN
>
> "SVM Returns" `1.52`:1 RETURN
>
> "Actual Returns" `1.31`:1 RETURN

Did this new model perform better or worse than your tuned trading 
algorithm?

### ANSWER 
# THE SELECTED LOGISTIC REGRESSION MODEL PERFORMED BEST IN OUR ANALYSIS.
> ![image.png](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_Returns_Final_Hour.png?raw=true)
>
> "LR Returns" `1.64`:1 RETURN
>
> "Tuned SVM Returns" `1.62`:1 RETURN
>
> "SVM Returns" `1.52`:1 RETURN
>
> "Actual Returns" `1.31`:1 RETURN
>

# THE LR MODEL OUTPERFORMED THE OTHER MODELS BY REACHING APPROX. 200% OF ACTUAL RETURNS IN Q4 OF 2020. FURTHER, IT'S THE ONLY MODEL THAT DIDN'T LOSE VALUE DURING THE BACKTEST FOR 2020.
>
> ![image.png](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_Returns_2020.png?raw=true)

---
---
---
---
### [Module 14 Challenge Submission](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/M14_Challenge_KonradK_machine_learning_trading_bot.ipynb)


![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_vs_Actual_Returns.png?raw=true)

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_Returns_Stable.png?raw=true)

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_Returns_Knotty.png?raw=true)

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_Returns_2020.png?raw=true)

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_Returns_Bottom.png?raw=true)

![image](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/Documentation/Images_14/plt_ALL_Returns_Final_Hour.png?raw=true)



---
### [Module 14 Challenge Submission](https://github.com/sfkonrad/M14_Challenge_Submission/blob/main/M14_Challenge_Submission/M14_Challenge_KonradK_machine_learning_trading_bot.ipynb)

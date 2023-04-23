# SC1015 Stock Prediction

## About
This is a [Stock Prediction Mini-Project](https://github.com/CodeLutetium/SC1015-Stock-Predictor/blob/main/Stock%20Prediction.ipynb) for SC1015 (Introduction to Data Science and Artificial Intelligence) which uses 2 datasets, the **Yahoo Finance AAPL Stock Price** data, and **Twitter Tweets** containing 'AAPL' or general stock market ticker symbols from **2015 to 2019**.

There are 3 main sections to our project:
1. [Sentiment Analysis of Tweets](https://github.com/CodeLutetium/SC1015-Stock-Predictor/tree/main/Sentimental%20Analysis)
2. [Technical Analysis of AAPL Stock](https://github.com/CodeLutetium/SC1015-Stock-Predictor/tree/main/Technical%20Indicator%20Models)
3. [LSTM Analysis of AAPL Stock Price](https://github.com/CodeLutetium/SC1015-Stock-Predictor/tree/main/LSTM%20Models)

Disclaimer:
**The notebooks are best viewed using JupyterNotebook or VisualStudioCode as our plotly charts are unable to render via github's online notebook view.**

## Contributors
- [@JustinWong645](https://github.com/JustinWong645) (Justin Wong) - Sentiment Analysis
- [@jaredpek](https://github.com/jaredpek) (Jared Pek) - Technical Analysis
- [@CodeLuteTium](https://github.com/CodeLutetium) (Mingyang) - LSTM Analysis

## Problem Definition
- Are we able to predict whether the price of a stock will rise or fall based on its forecasted price, technical indicators and tweets on the market conditions?
- Which of these 3 would be the best to predict it?

## Models Used
1. Sentiment Analysis
    - To Obtain Sentiment
        - Valence Aware Dictionary and Sentiment Reasoner
    - To Classify Rise or Fall
        - Linear Discriminant Analysis
        - **Random Forest Classifier**
        - Logistic Regression
2. Technical Analysis
    - To Classify Rise (Buy) or Fall (Sell)
        - Random Forest Classifier
        - Decision Tree Classifier
        - Stochastic Gradient Descent Classifier
        - **Logistic Regression**
3. LSTM Analysis
    - Long Short-Term Memory Networks

## Conclusion
1. Sentiment Analysis
    - RandomForestClassifier is the most reliable model with 56% accuracy without any extreme bias.
2. Technical Analysis
    - LogisticRegression was the best out of the 4 models to classify buy and sell signals based on provided technical indicators.
    - LogisticRegression had a very high prediction accuracy of 0.956, which demonstrates its effective prediction ability.
    - Tree-based models have been demonstrated to be more prone to overfitting.
3. LSTM Analysis
    - The 100 days LSTM training window had the lowest mean squared error in predicting future stock price.
4. Overall
    - The 3 methods to predict stock price movement are effective in helping us make buy or sell decisions for AAPL stock. 
    - However, individual methods are far from perfect in accurately predicting what happens in the stock market. Hence, it is important to combine and utilise all 3 methods in order to make an informed decision.

## What did we learn from this project?
1. Sentiment Analysis
    - Data extraction from a large dataset.
    - Data cleaning techniques using python's pandas library.
    - Sentiment Analysis using VADER.
    - Selection of which sentiment variable is best to use.
    - A highly speculative crowd online proved it hard to use people's sentiments to come up with highly accurate prediction.
    - Sentiment analysis is best used with other indicators to help investors come up with an informed decision.
2. Technical Analysis
    - Data extraction from online stock APIs.
    - Data cleaning techniques using python's pandas library.
    - Data visualisation using the plotly library, which generates interactive and responsive charts.
    - Calculation of various technical indicators like SMA, EMA, MACD and RSI from the closing price of stocks.
    - A model that could perform very well in theory, like our baseline RandomForestClassifier model, may not necessarily be the best model, hence we must experiment and explore different models to find the best model for our use.
    - Training, Tuning and Testing the RandomForestClassifier, DecisionTreeClassifier, SGDClassifier and LogisticRegression classification models using the scikit-learn library.
    - Hyperparameter tuning does not always improve a model, and the default untuned model could perform better than the same tuned model.
    - Different models have different decision making processes, hence will utilise different features of the dataset to make its decision.
3. LSTM Analysis
    - The greater the training window, the longer the training time for the model, and the lower the uncertainty and the more accurate the model is in predicting future price.
    - With a greater window, there is higher chance of overfitting, so we must carefully select our N value to strike a balance between prediction accuracy and fitting of our model on the dataset.

## References
1. Sentiment Analysis
    - [LinearDiscriminantAnalysis Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
    - [LogisticRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    - [RandomForestClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
2. Technical Analysis
    - [SMA and EMA Calculation](https://medium.com/codex/simple-moving-average-and-exponentially-weighted-moving-average-with-pandas-57d4a457d363#:~:text=SMA%20can%20be%20implemented%20by,average%20over%20a%20fixed%20window.&text=Where%20the%20window%20will%20be,used%20for%20calculating%20the%20statistic.)
    - [MACD Calculation](https://www.learnpythonwithrune.org/pandas-calculate-the-moving-average-convergence-divergence-macd-for-a-stock/)
    - [RSI Calculation](https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/)
    - [RandomForestClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    - [DecisionTreeClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    - [SGDClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
    - [LogisticRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
3. LSTM Analysis
    - [LSTM Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)

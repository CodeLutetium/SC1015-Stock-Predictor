# SC1015 Stock Prediction

## About

This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which uses 2 datasets, the Yahoo Finance AAPL Stock Price data, and Twitter Tweets containing 'AAPL' or general stock market ticker symbols from 2015 to 2019. 

There are 3 main sections to this projects.
1. <a href='https://github.com/CodeLutetium/SC1015-Stock-Predictor/tree/main/LSTM%20models'>LSTM Analysis of AAPL Stock Price</a>
2. <a href='https://github.com/CodeLutetium/SC1015-Stock-Predictor/tree/main/Technical%20Indicator%20Models'>Technical Analysis of AAPL Stock</a>
3. <a href='https://github.com/CodeLutetium/SC1015-Stock-Predictor/tree/main/Sentimental%20Analysis'>Sentiment Analysis of Tweets</a>

## Contributors

- @CodeLuteTium (Mingyang) - LSTM Analysis
- @jaredpek (Jared Pek) - Technical Analysis
- @JustinWong645 (Justin Wong) - Sentiment Analysis

## Problem Definition

- Are we able to predict whether the price of a stock will rise or fall based on its forecasted price, technical indicators and tweets on the market conditions?
- Which of these 3 would be the best to predict it?

## Models Used

1. LSTM Analysis
    - Long Short-Term Memory Networks
2. Technical Analysis
    - To Classify Rise (Buy) or Fall (Sell)
        - Random Forest Classifier
        - Decision Tree Classifier
        - Stochastic Gradient Descent Classifier
        - **Logistic Regression**
3. Sentiment Analysis
    - To Obtain Sentiment
        - Valence Aware Dictionary and Sentiment Reasoner
    - To Classify Rise or Fall
        - Linear Discriminant Analysis
        - **Random Forest Classifier**
        - Logistic Regression


## Conclusion

1. LSTM Analysis
2. Technical Analysis
    - LogisticRegression was the best out of the 4 models to classify buy and sell signals based on provided technical indicators.
    - LogisticRegression had a very high prediction accuracy of 0.956, which demonstrates its effective prediction ability.
    - Tree-based models have been demonstrated to be more prone to overfitting.
3. Sentiment Analysis
4. Overall
    - The 3 methods to predict stock price movement are effective in helping us make buy or sell decisions for AAPL stock. 
    - However, individual methods are far from perfect in accurately predicting what happens in the stock market. Hence, it is important to combine and utilise all 3 methods in order to truly make an informed decision.

## What did we learn from this project?

1. LSTM Analysis
2. Technical Analysis
    - Data extraction from online APIs.
    - Data cleaning techniques using python's pandas library.
    - Calculation of SMA, EMA, MACD and RSI values from the closing price of stocks.
    - A model that could perform very well in theory, like our baseline RandomForestClassifier model may not necessarily be the best model, hence we must experiment and explore different models to find the best one for our use.
    - Training, Tuning and Testing the RandomForestClassifier, DecisionTreeClassifier, SGDClassifier and LogisticRegression classification models using the scikit-learn library.
    - Hyperparameter tuning does not always improve a model, and the default untuned model could perform better than the same tuned model.
3. Sentiment Analysis

## References

1. LSTM Analysis
2. Technical Analysis
    - <a href='https://medium.com/codex/simple-moving-average-and-exponentially-weighted-moving-average-with-pandas-57d4a457d363#:~:text=SMA%20can%20be%20implemented%20by,average%20over%20a%20fixed%20window.&text=Where%20the%20window%20will%20be,used%20for%20calculating%20the%20statistic.'>SMA and EMA Calculation</a>
    - <a href='https://www.learnpythonwithrune.org/pandas-calculate-the-moving-average-convergence-divergence-macd-for-a-stock/'>MACD Calculation</a>
    - <a href='https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/'>RSI Calculation</a>
    - <a href='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'>RandomForestClassifier Documentation</a>
    - <a href='https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html'>DecisionTreeClassifier Documentation</a>
    - <a href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html'>SGDClassifier Documentation</a>
    - <a href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'>LogisticRegression Documentation</a>
3. Sentiment Analysis
    - <a href='https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html'>LinearDiscriminantAnalysis Documentation</a>
    - <a href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'>LogisticRegression Documentation</a>
    - <a href='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'>RandomForestClassifier Documentation</a>

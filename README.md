# Yes-Bank-Stock-Price-ML-Project
Yes Bank Stock Price analysis by Regression Machine Learning

![download (39)](https://github.com/user-attachments/assets/e339f6ff-8d1e-4ed4-b06c-0e1e7cc36b98) 

# Project Summary
In the domain of financial forecasting, the ability to predict stock prices is paramount for investors and analysts alike. This study focuses on utilizing regression analysis techniques to forecast the closing price of Yes Bank stock. A diverse array of regression models has been developed and meticulously evaluated to pinpoint the most precise predictive model. These models encompass Linear Regression, Ridge Regression, Lasso Regression, Random Forest Regression, Gradient Boosting Regression, and Support Vector Regression.

The performance of each model was assessed using key evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared. Additionally, feature importance analysis techniques were applied, including coefficient analysis, feature importance plots, and permutation feature importance, to unravel the significance of various features in predicting Yes Bank's closing price.

In addition to evaluating model performance, we explore feature importance to understand the factors driving stock price movements. This involves analyzing the coefficients of the regression models to identify the most influential variables. Furthermore, we utilize techniques such as feature selection algorithms, and permutation importance to gain deeper insights into the relative importance of each feature.

Following rigorous evaluation and analysis, the most accurate regression model for forecasting Yes Bank's closing stock price was identified. This comprehensive methodology not only facilitates the anticipation of future stock prices but also furnishes invaluable insights into the underlying drivers influencing Yes Bank's performance in the financial markets. Ultimately, leveraging regression analysis empowers stakeholders to make well-informed decisions within the dynamic landscape of stock market investments.

# Problem Statement
The primary challenge in this project is to construct a machine learning model capable of accurately forecasting the closing price of Yes Bank stocks. This necessitates training the model on a subset of historical data and validating its performance on a separate subset to ensure its reliability in predicting future stock prices. The ultimate goal is to develop a robust and precise predictive model that can offer valuable insights into the behavior of Yes Bank's stock market by accurately forecasting its closing prices.

# Dataset 
The dataset used for this project consists of historical stock data for Yes Bank, including features such as open price, high price, low price, and  most importantly, closing price. This data is essential for training and evaluating regression models.

# Methods Used

1. **Data Preprocessing** Data preprocessing involves handling missing values, removing outliers, and ensuring data quality. Additionally, we will split the data into training and testing sets for model development and evaluation.

2. **Feature Engineering** Feature engineering is a critical step where we create meaningful features that can influence the closing price prediction. It includes selecting relevant columns, transforming data, and creating new features when necessary.

3. **Regression Models** Several regression models will be implemented and evaluated, including Linear Regression, Linear Regression using Lasso Regularization, Linear Regression with Ridge Regularization, Linear Regression with Elastic Net Regularization. We will assess their performance and select the most accurate model.

4. **Evaluation Model** evaluation will be based on various regression metrics to determine the model's accuracy in predicting Yes Bank's closing prices. Common metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), R-squared (R2) and Adjusted R-squared (R2).

5. **Results** The results section will present the findings of the regression analysis, including the performance of different models and their predictions for Yes Bank's closing prices.

# Random Forest

![Random-Forest-Algorithm-1024x576](https://github.com/user-attachments/assets/7621f9eb-6df7-4e2e-b42e-28ac2419afa7)

 Random Forest builds multiple decision trees from bootstrap samples of the original dataset, introducing randomness by selecting a random subset of features at each split. This technique helps create less     
 correlated trees, enhancing the model's robustness and predictive performance. In classification, the final output is determined by majority voting, where each tree votes for a class, and the class with the most 
 votes is chosen. In regression, the final prediction is the average of all the trees' outputs. This approach reduces the risk of overfitting, a common issue with single decision trees, and generally improves 
 accuracy.

 # XGBoost
 ![img-3](https://github.com/user-attachments/assets/ae3db65c-2a57-4136-9184-7631d6f234b0) 
 XGBoost (eXtreme Gradient Boosting) is an advanced implementation of the gradient boosting technique for supervised learning tasks, known for its speed and performance. It builds an ensemble of trees sequentially, where each new tree attempts to correct the errors made by the previous ones. This is achieved by optimizing a specific loss function and adding new trees to model the residuals of the predictions. XGBoost incorporates several enhancements to improve its efficiency and effectiveness, such as regularization to prevent overfitting, a weighted quantile sketch algorithm for handling sparse data, and parallel processing capabilities to speed up training. These features make XGBoost particularly suitable for large-scale datasets and high-dimensional data. Additionally, XGBoost includes functionalities for early stopping, missing value handling, and cross-validation. It is widely used in machine learning competitions and practical applications due to its ability to deliver high predictive performance with relatively low computational cost.

 # Conclusion 
 Our first and base model was Linear regression, and got a decent training and testing score of 0.82 and 0.76 respectively with high MSE and MAE metric score.

Then we implemented Ridge, lasso and ElasticNet regularization models and the Lasso model was found to as a better model with 0.83 r2-score and MSE and MAE was also very low as compared to base model.

Then we tried Polyfit model which was giving a pretty good result but was performing very bad in cross-validation.

Then we implemented Random Forest Regressor and XGboost Regressor, both of them were performing far better than other previous models at every metric..


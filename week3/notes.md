## Week 3 Notes : Classification

### Video 1 : Churn Prediction Project

----
[![Churn Prediction Project](https://img.youtube.com/vi/0Zw04wdeTQo/0.jpg)](https://www.youtube.com/watch?v=0Zw04wdeTQo&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=29)
- Telecom customer churn prediction project; Determine if a customer will leave the service or not
- Model will predict the customer churn

### Video 2 : Data Preparation

----
[![Data Preparation](https://img.youtube.com/vi/VSGGU9gYvdg/0.jpg)](https://www.youtube.com/watch?v=VSGGU9gYvdg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=30)

- Download CSV file from [zoomcamp](https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv)
- Change the column names from spaces to underscore
- change the object type values from space to underscore
- Fix `totalcharges` column to numeric from object
- Modify `churn` column from boolean to int 

### Video 3 : Setting Up The Validation Framework

----
[![Setting Up The Validation Framework](https://img.youtube.com/vi/_lwz34sOnSE/0.jpg)](https://www.youtube.com/watch?v=_lwz34sOnSE&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=31)

- Split the data into train and test sets using `train_test_split` from sklearn
- Get the y variable from the train and test sets using the `churn` column'
- Drop the target variable from the x sets

### Video 4 : EDA

----
[![EDA](https://img.youtube.com/vi/BNF1wjBwTQA/0.jpg)](https://www.youtube.com/watch?v=BNF1wjBwTQA&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=32)

- Understand the split of the target variable using `value_counts` and `mean`
- Create numerical and categorical columns list 
- Understand the unique values in the categorical columns using `nunique`


### Video 5 : Feature Importance: Churn Rate And Risk Ratio

----
[![Feature Importance: Churn Rate And Risk Ratio](https://img.youtube.com/vi/fzdzPLlvs40/0.jpg)](https://www.youtube.com/watch?v=fzdzPLlvs40&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=33)

- Calculation of churn rate and risk ratio for categorical columns
- Churn Rate = (Number of churned customers in a category) / (Total number of customers in that category)
- Risk Ratio = (Churn Rate of a category) / (Overall Churn Rate)
- IPython display is used to display the results in a tabular format when inside a loop

### Video 6 : Feature Importance: Mutual Information

----
[![Feature Importance: Mutual Information](https://img.youtube.com/vi/_u2YaGT6RN0/0.jpg)](https://www.youtube.com/watch?v=0kYk2bX4HhA&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=34)

- Mutual Information - concept from information theory, it tells how much we can learn about one variable by knowing the value of another
- sklearn `mutual_info_score` function can be used to calculate mutual information between categorical features and target variable
```
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

# calculate mutual information for each categorical variable
mi = df_full_train[categorical].apply(mutual_info_churn_score)

# print the results sorted by value to understand the important features impacting target variable 'churn'
mi.sort_values(ascending=False)
```

### Video 7 : Feature Importance: Correlation

----
[![Feature Importance: Correlation](https://img.youtube.com/vi/mz1707QVxiY/0.jpg)](https://www.youtube.com/watch?v=mz1707QVxiY&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=35)

- Correlation is used to understand the relationship between numerical features and target variable
- Pearson Correlation
    - Ranges from -1 to 1
- Correlation Interpretation
    - -1 : Perfect Negative Correlation
    - 0 : No Correlation
    - 1 : Perfect Positive Correlation
- more correlation interpretation
    - 0.9 to 1 (-0.9 to -1) : Very Strong (Negative) Correlation
    - 0.7 to 0.9 (-0.7 to -0.9) : Strong (Negative) Correlation
    - 0.5 to 0.7 (-0.5 to -0.7) : Moderate (Negative) Correlation
    - 0.3 to 0.5 (-0.3 to -0.5) : Weak (Negative) Correlation
    - 0 to 0.3 (0 to -0.3) : Negligible (Negative) Correlation

### Video 8 : One-Hot Encoding

----
[![One-Hot Encoding](https://img.youtube.com/vi/L-mjQFN5aR0/0.jpg)](https://www.youtube.com/watch?v=L-mjQFN5aR0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=36)

- gender and contract table with random values

| Gender | Contract |
|--------|----------|
| F      | 2Y       |
| M      | 1Y       |
| F      | MTH      |
| F      | 1Y       |
| M      | MTH      |
| F      | 2Y       |

- Will transform to the following when one-hot encoded, by transforming each categorical variable into multiple binary variables

| F | M | MTH | 1Y | 2Y |
|---|---|-----|----|----|
|   |   |     |    |    |
| 1 | 0 | 0   | 0  | 1  |
| 0 | 1 | 0   | 1  | 0  |
| 1 | 0 | 1   | 0  | 0  |
| 1 | 0 | 0   | 1  | 0  |
| 0 | 1 | 1   | 0  | 0  |
| 1 | 0 | 0   | 0  | 1  |

- `DictVectorizer` in scikit learn can be used for one-hot encoding
- `dv.fit_transform` is used to fit and transform the training data
- `dv.transform` is used to transform the validation and test data

### Video 9 : Logistic Regression

----
[![Logistic Regression](https://img.youtube.com/vi/7KFE2ltnBAg/0.jpg)](https://www.youtube.com/watch?v=7KFE2ltnBAg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=37)

- logistic regression is used for solving binary classification problems
- '0' is for negative class and '1' is for positive class
- `g(Xi) -> {0,1}`
- `g(Xi) = sigmoid(W0 + W^T * Xi)`
    - Output of Logistic Regression is between 0 & 1
- 'z' on x-axis is called logit and sigmoid on y-axis is called logistic function
    - max value for z is +infinity and min value is -infinity
    - max value for logistic function is 1 and min value is 0

### Video 10 :  Training Logistic Regression with Scikit-Learn

----
[![Training Logistic Regression with Scikit-Learn](https://img.youtube.com/vi/hae_jXe2fN0/0.jpg)](https://www.youtube.com/watch?v=hae_jXe2fN0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=38)

- Create the LogisticRegression model using scikit `LogisticRegression`
- `model.coef_` gives the weights for each feature
- `model.predict(X_train)` gives the predicted class labels
- `model.predict_proba(X_train)` gives the predicted probabilities for each class
- `churn_decision = y_pred > 0.5` calculates the boolean 
- Accuracy is calculated to evaluate the model performance
  - `(y_val == churn_decision).mean()`

### Video 11 : Model Interpretation

----
[![Model Interpretation](https://img.youtube.com/vi/OUrlxnUAAEA/0.jpg)](https://www.youtube.com/watch?v=OUrlxnUAAEA&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=39)

- Subset of the features were taken and a smaller model is built. This was used to interpret the model better by looking at the intercept and weights
- `model.intercept_` gives the intercept value
- `model.coef_` gives the weights for each feature
- `sigmoid(_)` function is defined to calculate the sigmoid value

### Video 12 : Using the model

----
[![Using the model](https://img.youtube.com/vi/Y-NGmnFpNuM/0.jpg)](https://www.youtube.com/watch?v=Y-NGmnFpNuM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=40)

- Model created from full train and val data
- Tested on test data to check customer churn

### Video 13 : Summary

----
[![Summary](https://img.youtube.com/vi/Zz6oRGsJkW4/0.jpg)](https://www.youtube.com/watch?v=Zz6oRGsJkW4&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=41)


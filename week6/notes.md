## Week 6 Notes : Trees

Practice jupyter notebooks: 
   1. [Trees](week6-trees.ipynb)

### Video 1 :  Credit Risk Scoring Project 

----

[![Intro / Credit Risk Scoring Project ](https://img.youtube.com/vi/GJGmlfZoCoU/0.jpg)](https://www.youtube.com/watch?v=GJGmlfZoCoU&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=59)

- Goal is to build a model that Banks can use to decide whether to give a loan to a customer or not
- The decision of the model will be based on the credit risk score using the customer's information, historical data and their payback behavior
- Dataset used : [credit scoring data](https://github.com/gastonstat/CreditScoring)

### Video 2 :  Data Cleaning and Preparation

----

[![Data Cleaning and Preparation](https://img.youtube.com/vi/tfuQdI3YO2c/0.jpg)](https://www.youtube.com/watch?v=tfuQdI3YO2c&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=60)

- Data downloaded and analysed
- Following are done
  1. Categorical mapped to string values
  2. Missing values handled
  3. Unknown values handled
  4. Data split into train/val/test
- Categorical values are marked as 0,1,2 etc., need to be converted to corresponding string values
- File in the [source R](https://github.com/gastonstat/CreditScoring/blob/master/Part1_CredScoring_Processing.R) file has the mapping of categorical values
```
levels(dd$Status) = c("good", "bad")
levels(dd$Home) = c("rent", "owner", "priv", "ignore", "parents", "other")
levels(dd$Marital) = c("single", "married", "widow", "separated", "divorced")
levels(dd$Records) = c("no_rec", "yes_rec")
levels(dd$Job) = c("fixed", "partime", "freelance", "others")
```

### Video 3 :  Decision Trees

----

[![Decision Trees](https://img.youtube.com/vi/YGiQvFbSIg8/0.jpg)](https://www.youtube.com/watch?v=YGiQvFbSIg8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=61)

- Following is a sample decision tree (in ascii format for the credit risk data)
```
        [records = "yes"]
        /                \
      False             True
      /                     \
   [Assets > 6000]     [Job = "parttime"]
    /      \                /       \
  False   True           False       True
  /         \            /             \
["Default"] ["ok"]      ["Ok"]      ["Default"]
      
```

- `DecisionTreeClassifier` from `sklearn.tree` is used to build decision tree models
- Decision trees can overfit the data
  - Overfitting means that the model performs well on training data but poorly on unseen data
  - Model fails to generalize
- Depth of the tree is one way to control overfitting. It is also referred as `max_depth` hyperparameter
- Tree with max depth as a 1 is called as `Decision Stump`
- `export_text(dt, feature_names=list(dv.get_feature_names_out()))` does the ascii representation of the decision tree
- `records_no`
  - `records=no <= 0.50` means Records is 'present'
  - `records=no > 0.50` means Records is 'not present'
- `class:
  - 1 : default
  - 0 : ok

### Video 4 : Decision Tree Learning Algorithm

----

[![Decision Tree Learning Algorithm](https://img.youtube.com/vi/XODz6LwKY7g/0.jpg)](https://www.youtube.com/watch?v=XODz6LwKY7g&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=62)


- Determining the point of best split using a sample of asset value and default status 
- mis-classification error is calculated for different split points

- A table like the one below could help on this example of one split
- Split on Assets > T
  | T    | Decision Left | Impurity Left | Decision Right | Impurity Right | Average |
|------|---------------|---------------|----------------|----------------|---------|
| 0    | Default       | 0%            | Ok             | 43%            | 21%     |
| 2000 | Default       | 0%            | Ok             | 33%            | 16%     |
| 3000 | Default       | 0%            | Ok             | 20%            | 10%     |
| 4000 | Default       | 25%           | Ok             | 25%            | 25%     |
| 5000 | Default       | 50%           | Ok             | 50%            | 50%     |
| 8000 | Default       | 43%           | Ok             | 0%             | 21%     |

- Here the split at 3000 has the lowest average impurity of 10%
- We can add more features like for e.g., debt and calculate the impurity for different split points
```python
thresholds = {
    'assets': [0, 2000, 3000, 4000, 5000, 8000],
    'debt': [500, 1000, 2000]
}
```

### Video 5 :  Decision Trees Parameter Tuning

----

[![Decision Trees Parameter Tuning](https://img.youtube.com/vi/XJaxwH50Qok/0.jpg)](https://www.youtube.com/watch?v=XJaxwH50Qok&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=63)

- selecting `max_depth`  : Maximum depth of the tree
- selecting `min_samples_leaf` : Minimum number of samples that are to be present in a leaf node
- Iterate over different combinations of hyperparameters to find the best model
```python
scores = []

for depth in [4, 5, 6]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((depth, s, auc))
```

### Video 6 :  Ensemble Learning and Random Forest

----

[![Ensemble Learning and Random Forest](https://img.youtube.com/vi/FZhcmOfNNZE/0.jpg)](https://www.youtube.com/watch?v=FZhcmOfNNZE&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=64)

- Ensemble learning is a technique where multiple models (weak learners) are combined to create a stronger model
- Multiple decision trees are trained and their predictions are aggregated to make a final prediction
- Bagging is the technique used in Random Forest where multiple subsets of data are created by sampling with replacement
- Each tree gets features randomly selected from the feature set and data points randomly selected from the training data
- `RandomForestClassifier` from `sklearn.ensemble` is used to build Random Forest
- Hyper parameters for RandomForestClassifier used
  - `n_estimators` : Number of trees in the forest
  - `max_depth` : Maximum depth of the tree
  - `min_samples_leaf` : Minimum number of samples that are to be present in a leaf node
- Iterate over different combinations of hyperparameters to find the best model by finding the AUC score on validation data
- plot them and visualize the results

### Video 7 :  Gradient Boosting and XGBoost

----

[![Gradient Boosting and XGBoost](https://img.youtube.com/vi/xFarGClszEM/0.jpg)](https://www.youtube.com/watch?v=xFarGClszEM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=65)

- Boosting is the technique where models are trained sequentially, each trying to correct the errors of the previous one
- XGBoost is an optimized implementation of gradient boosting
- `XGBClassifier` from `xgboost` library is used to build XGBoost models

## Video 8 : XGBoost Parameter Tuning

----

[![XGBoost Parameter Tuning](https://img.youtube.com/vi/VX6ftRzYROM/0.jpg)](https://www.youtube.com/watch?v=VX6ftRzYROM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=66)

- 3 Hyper parameters are tunes
  - eta : Learning rate i.e., how much percentage of correction is applied from previous model
  - max_depth : Maximum depth of the tree, i.e., how many splits are allowed
  - min_child_weight : Minimum sum of instance weight needed in a child i.e., minimum number of samples required in a leaf node

### Video 9 :  Select the Best Model

----

[![Select the Best Model](https://img.youtube.com/vi/lqdnyIVQq-M/0.jpg)](https://www.youtube.com/watch?v=lqdnyIVQq-M&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=67)
- 
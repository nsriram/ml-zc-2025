## Week 4 Notes : Evaluation
Practice jupyter notebook: [Evaluation Metrics](week4-evaluation.ipynb)


### Video 1 : Evaluation Metrics: Session Overview

----
[![Evaluation Metrics: Session Overview](https://img.youtube.com/vi/gmg5jw1bM8A/0.jpg)](https://www.youtube.com/watch?v=gmg5jw1bM8A&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=42)

- Same model is used from week 3 (logistic regression)

### Video 2 : Accuracy and dummy model

----

[![Accuracy and dummy model](https://img.youtube.com/vi/FW_l7lB0HUI/0.jpg)](https://www.youtube.com/watch?v=FW_l7lB0HUI&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=43)

- `accuracy_score` used for calculating accuracy
- Multiple thresholds can give different accuracy values and is evaluated 
- It is observed that for 0.5 threshold, accuracy is 0.80 and was the highest
- The provided data has class imbalance 
  - Accuracy could be high in a class imbalanced dataset by predicting the majority class all the time; 
  - Here in this case it could predict that all customers will not churn and get 80% accuracy
  - Hence, accuracy as a score is not a good metric in such cases

### Video 3 : Confusion Table

----

[![Confusion Table](https://img.youtube.com/vi/Jt2dDLSlBng/0.jpg)](https://www.youtube.com/watch?v=Jt2dDLSlBng&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=44)

- Way of looking at different errors and correct predictions in a classification model
- This is viewed using a 2x2 table

#### Churn logic
Given `t` as threshold and `g(x)` as predicted probability of churn for customer `x`:
- `g(x) >= t` => predict churn
- `g(x) < t` => predict no churn

#### Prediction possibilities
1. **True Positive** : Model predicted customer will churn and they did
2. **False Positive** : Model predicted customer will churn but they didn't
3. **False Negative** : Model predicted customer will not churn but they did
4. **True Negative** : Model predicted customer will not churn and they didn't

#### Confusion Matrix Format
|                     | Predicted Negative  | Predicted Positive  |
|---------------------|---------------------|---------------------|
| Actual Negative     | True Negative (TN)  | False Positive (FP) |
| Actual Positive     | False Negative (FN) | True Positive (TP)  |


### Video 4 : Precision and Recall

----

[![Precision and Recall](https://img.youtube.com/vi/gRLP_mlglMM/0.jpg)](https://www.youtube.com/watch?v=gRLP_mlglMM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=45)

Different metrics derived from confusion matrix

1. Accuracy = (TP + TN) / (TP + TN + FP + FN)
2. Precision = TP / (TP + FP)
3. Recall = TP / (TP + FN)
4. F1Score = 2 * (Precision * Recall) / (Precision + Recall)

- Precision : out of all predicted positives, how many are actually positive
- Recall : out of all actual positives, how many are predicted positive
- For the model that has data with class imbalance, precision is 0.67 and recall is 0.55


### Video 5 : ROC Curves

----

[![ROC Curves](https://img.youtube.com/vi/dnBZLk53sQI/0.jpg)](https://www.youtube.com/watch?v=dnBZLk53sQI&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=46)

- ROC abbreviates to Receiver Operating Characteristic
- False Positive Rate (FPR) = FP / (FP + TN) i.e., False Positive divided by Actual Negatives
- True Positive Rate (TPR) = TP / (TP + FN) = Recall i.e., True Positive divided by Actual Positives
- ROC curve plots TPR vs FPR at different threshold values
  - TPR, FPR compared against random, ideal models
  - Reasonably good model is between the random and ideal models
  - For ideal models the FPR is 0 and TPR is 1
  - if the curve goes below the random model, then the model is worse than random and can be inverted to get a better model
- scikit function `roc_curve` can be used to get the TPR, FPR values for different thresholds

### Video 6 : ROC - Area Under the Curve (AUC)

----
[![ROC - Area Under the Curve (AUC)](https://img.youtube.com/vi/hvIQPAwkVZo/0.jpg)](https://www.youtube.com/watch?v=hvIQPAwkVZo&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=48)

- Area under the ROC curve is called AUC
- AUC ranges from 0 to 1
- AUC can indicate how good the model is
- baselines
  - for random model, AUC is around 0.5
  - for ideal model, AUC is 1
  - for the created model the AUC could be in the 0.7-0.8 range and is considered a good model
- `auc` function from scikit can be used to calculate AUC from TPR and FPR values
- `roc_auc_score` function from scikit can be used to calculate AUC directly from actual and predicted values

### Video 7 : Cross-Validation

----

[![Cross-Validation](https://img.youtube.com/vi/BIIZaVtUbf4/0.jpg)](https://www.youtube.com/watch?v=BIIZaVtUbf4&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=49)

- Cross-validation is a technique to evaluate model performance
- Split dataset into k folds
- For each fold
  - Use the fold as test set
  - Use remaining folds as training set
  - Train model on training set
  - Evaluate model on test set
  - Collect evaluation metrics
- Average the metrics across all folds to get overall performance estimate
- `cross_val_score` function from scikit can be used to perform cross-validation easily
#!/usr/bin/env python

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from custom_transformer import PhZeroToNaN

# print versions of libraries
print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

# load data
def load_data():
    data_file = "water_potability.csv"
    df = pd.read_csv(data_file)
    return df


def train_model(df):
    num_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
                    'Trihalomethanes', 'Turbidity']
    y_train = df.Potability
    train_dict = df[num_features]

    # Preprocessing: ph fix + median impute
    preprocess = make_column_transformer(
        (Pipeline([
            ("ph_fix", PhZeroToNaN(ph_col="ph")),
            ("impute", SimpleImputer(strategy="median"))
        ]), num_features),
        remainder='drop'
    )

    # create the base Random Forest classifier model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Make pipeline using the preprocess and rf base model
    rf_pipeline = make_pipeline(preprocess, rf)

    # declare the param grid with hyperparameter combinations
    param_grid = {
        "randomforestclassifier__n_estimators": [200, 400, 600],
        "randomforestclassifier__max_depth": [None, 10, 20],
        "randomforestclassifier__min_samples_split": [2, 3, 5],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4],
        "randomforestclassifier__max_features": ["sqrt", "log2", None],
        "randomforestclassifier__class_weight": [None, "balanced"]
    }

    # GridSearchCV without explicit StratifiedKFold (cv=5 -> KFold)
    model_grid = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    # Fit the grid search on the training data
    model_grid.fit(train_dict, y_train)
    return model_grid

#load data
data_frame = load_data()
print("Data loaded successfully.")

# train the model
trained_model_grid = train_model(data_frame)
print("Model trained successfully.")

# save the model
final_model = trained_model_grid.best_estimator_
print("Model best estimator extracted.")

# Define the file path to save (serialize) the trained model along with the data preprocessing steps
saved_model_path = "water_potability_prediction_model_v1_0.joblib"
joblib.dump(final_model, saved_model_path)
print(f'Model saved to {saved_model_path} successfully.')

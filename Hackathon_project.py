#installing packages for machine learning and data analysis
import os
import sys
#data analysis tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def clean_data():
    data = pd.read_csv('train.csv')
    #handle missing values - fill with mean for numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].mean())
    
    return data

def train_model(data):
    X = data.drop(['id', 'overqualified'], axis=1)
    y = data['overqualified']
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Use one-hot encoding for nominal features
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning using GridSearchCV
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f'Best Parameters: {grid_search.best_params_}')
    
    model.fit(X_train, y_train)


    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    # enable cross validation for better model evaluation

    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f'Cross-validation scores: {cv_scores}')    
    return model



def main():
    data = clean_data()
    model = train_model(data)

main()



# Future improvements:
# hyperparameter tuning using grid search or random search
# cross validation for better model evaluation


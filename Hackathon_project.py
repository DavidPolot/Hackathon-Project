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
from sklearn.preprocessing import LabelEncoder

#Machine learning tools
import ultralytics
import tensorflow as tf
import torch as pt
import keras as ks 

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    return model

def main():
    data = clean_data()
    model = train_model(data)

main()





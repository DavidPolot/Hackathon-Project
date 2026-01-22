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

#Machine learning tools
import ultralytics
import tensorflow as tf
import pytorch as pt
import keras as ks 


#create frame for reading data before the hackathon project
data = pd.read_csv('data.csv')
print(data.head())
#preprocess data
data.fillna(method='ffill', inplace=True)
X = data.drop('target', axis=1)
y = data['target']
#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


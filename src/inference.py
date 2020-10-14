from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np


# Load model from file
classifer = joblib.load("model.pkl")

# Create new observation
new_observation = pd.read_csv("./newdata.csv")

# Predict observation's class
print(classifer.predict(new_observation))

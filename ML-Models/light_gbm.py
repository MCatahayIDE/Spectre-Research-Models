
# Implements Binary Classification Model using LightGBM trained on master dataset

##Imports

#pandas packages
import pandas as pd
import numpy as np
import lightgbm as lgb

#Metrics and Evaluation
import seaborn as sb
import matplotlib.pyplot as plt
#Classification report
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

#SKlearn packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#Import LightGBM Classifier
from lightgbm import LGBMClassifier

#Warnings and Deprecation
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


##Load Data from Data directory

try:
    # Assuming the CSV is in the same directory as the script.
    # If it's in a 'data' subdirectory, use 'data/combined_8k.csv'
    df = pd.read_csv('Data\master_combined_95k.csv')
except FileNotFoundError:
    print("Error: 'master_combined_95k.csv' not found. Please ensure the file path is correct.")
    exit()

#Load head and frame metrics to ensure data is loaded correctly

print("--- Data Head ---")
print(df.head())
print("\n--- Data Info ---")
df.info()

##Visualize Dataframe Metrics and Characteristics

#Implement
#Pie chart distribution of output binaries
#etc.


##Identify and Categorize Features and Data Output

#Define non-feature and target columns to exclude from model features
target_column_name = 'Attack_Cache_Binary'
target_col = df[target_column_name]
non_feature_col_key = [ 
    'index',
    'original_timestamp',
    'attack_ddos_binary',
    'attack_scan_binary',
    target_column_name
]

non_feature_col = [col for col in non_feature_col_key if col in df.columns]
features_col = [col for col in df.columns if col not in non_feature_col]

X_features = df[features_col]
#Define non-feature params


##Split Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(
X_features, target_col, test_size = 0.2, random_state = 50
)

##Standardize and scale data

#Initialize Standard Scaler
feature_scaler = StandardScaler()

#Fit and transform training data
feature_scaler.fit(X_train)
X_train = feature_scaler.transform(X_train)
X_test = feature_scaler.transform(X_test)

##Training and Predicting with LGBM Classifier

#Initialize LighGBM Classifier
lgbm_binary = LGBMClassifier(
    metric ='auc',
    random_state= 50
)

#Set hyperparameters for LightGBM
lgbm_binary.set_params(
    boosting_type='gbdt',
    objective='binary',
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)

#Train LightGBM Classifier on dataset set for training
lgbm_binary.fit(X_train, Y_train)

#Evaluate with predictions on test data
Y_pred = lgbm_binary.predict(X_test)
Y_pred_proba = lgbm_binary.predict_proba(X_test)[:, 1]  # Probability of the positive classs

##Metrics and Evaluation

#Print classification report
print("\n--- Classification Report ---")
# Corrected: Use Y_pred (model predictions) instead of X_test
# Added target_names for better readability of the report
print(classification_report(Y_test, Y_pred, target_names=['No Attack (0)', 'Attack (1)']))

# Accuracy Score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# ROC AUC Score
roc_auc = roc_auc_score(Y_test, Y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


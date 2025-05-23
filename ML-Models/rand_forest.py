import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
try:
    # Assuming the CSV is in the same directory as the script.
    # If it's in a 'data' subdirectory, use 'data/combined_8k.csv'
    df = pd.read_csv('Data\master_combined_95k.csv')
except FileNotFoundError:
    print("Error: 'master_combined_75k.csv' not found. Please ensure the file path is correct.")
    exit()

print("--- Data Head ---")
print(df.head())
print("\n--- Data Info ---")
df.info()

# --- 2. Identify Features (X) and Target (y) ---
target_column = 'Attack_Cache_Binary'

# Check if target column exists
if target_column not in df.columns:
    print(f"Error: Target column '{target_column}' not found in the CSV.")
    print(f"Available columns are: {df.columns.tolist()}")
    exit()

# Define columns that are NOT features.
# 'time_index' is likely an identifier.
# Other 'attack_*' columns are different labels or potentially redundant for this specific task.
# 'attack_no_attack' is likely the inverse of any attack occurring.
non_feature_columns = [
    'time_index',
    'index',
    target_column,
    'attack_no_attack', # If this is 1 when all other attacks are 0, it's highly correlated
    'attack_ddos_binary',
    'attack_scan_binary',
    'original_timestamp'
    # Add any other columns that are not hardware performance counters
]

# Filter out non-feature columns that might not exist in all CSV versions
actual_non_feature_columns = [col for col in non_feature_columns if col in df.columns]

feature_columns = [col for col in df.columns if col not in actual_non_feature_columns]

if not feature_columns:
    print("Error: No feature columns identified. Please check your 'non_feature_columns' list.")
    exit()

X = df[feature_columns]
y = df[target_column]

print(f"\n--- Target Variable: {target_column} ---")
print(y.value_counts(normalize=True)) # Check for class imbalance

# --- 3. Preprocessing ---
# Check for missing values
print("\n--- Missing Values Check ---")
missing_values = X.isnull().sum()
if missing_values.sum() > 0:
    print("Missing values found in features:")
    print(missing_values[missing_values > 0])
    # Simple imputation: fill with mean. For more complex scenarios, consider other strategies.
    X = X.fillna(X.mean())
    print("Missing values filled with mean.")
else:
    print("No missing values found in features.")

# --- 4. Split Data ---
# Stratify by y to ensure similar class proportions in train and test sets,
# especially important if the dataset is imbalanced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print(f"\nTraining set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Feature Scaling
# Random Forests are less sensitive to feature scaling, but it's good practice
# and doesn't hurt. It can also help with convergence for some internal calculations.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train Random Forest Model ---
# class_weight='balanced' can help with imbalanced datasets.
# You can tune hyperparameters like n_estimators, max_depth, etc. using GridSearchCV.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', oob_score=True)
print("\nTraining Random Forest model...")
rf_model.fit(X_train_scaled, y_train)
print(f"Model OOB Score: {rf_model.oob_score_:.4f}")

# --- 6. Make Predictions ---
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class (1)

# --- 7. Evaluate Model ---
print("\n--- Model Evaluation ---")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1-score (from classification report)
print("\nClassification Report:")
# Use target_names for better readability if you know what 0 and 1 represent
# For example: target_names=['No Attack (0)', 'Cache Attack (1)']
print(classification_report(y_test, y_pred, target_names=['No Attack (0)', 'Attack (1)']))


##
### Confusion Matrix and Graph Generation
##


# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No Attack (0)', 'Predicted Attack (1)'],
            yticklabels=['Actual No Attack (0)', 'Actual Attack (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# --- Feature Importances ---
print("\n--- Feature Importances ---")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_columns, # Use original feature names
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Top 10 Features:")
print(feature_importance_df.head(10))

# Plot top N feature importances
N_FEATURES_TO_PLOT = 15
plt.figure(figsize=(10, max(6, N_FEATURES_TO_PLOT * 0.4))) # Adjust height based on N
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(N_FEATURES_TO_PLOT), palette='viridis')
plt.title(f'Top {N_FEATURES_TO_PLOT} Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n--- Script Finished ---")

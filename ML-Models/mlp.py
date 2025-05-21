#define Multi-Layer Perceptron to be trained on combined data master CSVs

#pandas packages
import pandas as pd

#Scikit-learn packages
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#import data from Master-DF
try:
    #load CSV from Data
    principal_df = pd.read_csv('data\combined_8k.csv')
except FileNotFoundError:
    print("Error: 'combined_8k.csv' not found. Please ensure the file path is correct.")
    exit()

#load head to ensure data is loaded correctly
print("--- Data Head ---")
print(principal_df.head())
print("\n--- Data Info ---")
principal_df.info()

##Categorize Data and Target Output

#define target data column
target_col = 'Attack_Cache_Binary'
#@TODO Implement conditional for locating target value
##
##

##Define NON-feature columns
non_feature_col_key = [ 
    'index',
    'original_timestamp',
    'attack_ddos_binary',
    'attack_scan_binary',
    target_col
]


##Parse CSV for features and parse out non-features
non_feature_col = [col for col in non_feature_col_key if col in principal_df.columns]
feature_col = [col for col in principal_df.columns if col not in non_feature_col]

##Assign features and target after parsing
X_features = principal_df[feature_col]
Y_target = principal_df[target_col]

##Assign testing and training proportions to split dataset
X_train, X_test, Y_train, Y_test = train_test_split(

X_features, Y_target, test_size = 0.2

)

##Standardize and scale data
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)  

##Initialize MLP Binary Classifier Model
mlp_binary = MLPClassifier(
    hidden_layer_sizes = (256, 128, 64),
    activation = 'relu',
    max_iter = 1000,
    random_state = 50                      #seed training for reproducibility
)

##Train
mlp_binary.fit(X_train, Y_train)

##Post-training prediction on test data for evaluation
Y_pred = mlp_binary.predict(X_test)
accuracy_score = mlp_binary.score(X_test, Y_test)


##Print evaluation metrics
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score * 100, "%")

binary_class_report = classification_report(Y_test, Y_pred)
print ("\n--- Classification Report ---")
print(binary_class_report)


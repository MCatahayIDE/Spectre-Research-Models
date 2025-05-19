#define Multi-Layer Perceptron to be trained on combined data master CSVs
#pandas packages
import pandas as pd

#Scikit-learn packages
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#import data from Master-DF
principal_df = pd.read_csv('Master-DF.csv')

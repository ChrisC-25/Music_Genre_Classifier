from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from preprocess import return_data

# Load data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_data()

# Perform Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 100)

rf_classifier.fit(X_train_scaled, y_train)
y_pred = rf_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

class_matrix = classification_report(y_test, y_pred)
# print(class_matrix)

def get_random_forest_acc():
    return accuracy, conf_matrix

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from get_data import get_training_and_test_labels
from pca import get_pca_features

y_train, y_test = get_training_and_test_labels()
X_train_pca, X_test_pca = get_pca_features()

# Perform Naive Bayes
nb_classifier = GaussianNB()

nb_classifier.fit(X_train_pca, y_train)
y_pred = nb_classifier.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ")
print(accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
                           
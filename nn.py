import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from preprocess import return_data

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Load data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_data()

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train_encoded, epochs=100, validation_split=0.2)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
accuracy = accuracy_score(y_test_encoded, y_pred)
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

class_matrix = classification_report(y_test_encoded, y_pred)
# print(class_matrix)

def get_nn_acc():
    return history , accuracy, conf_matrix


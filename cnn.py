import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
 

from preprocess import return_data

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils import to_categorical


# Load data
X_train, X_test, X_val, y_train, y_test, y_val = return_data()


# Encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Model
model = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(15, activation='softmax') 
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#Train
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
y_test_decoded = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test_decoded, y_pred)
conf_matrix = confusion_matrix(y_test_decoded, y_pred)

class_matrix = classification_report(y_test_decoded, y_pred)
# print(class_matrix)

def get_cnn_acc():
    return history, accuracy, conf_matrix





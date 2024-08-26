from nn import get_nn_acc
from cnn import get_cnn_acc
from random_forest import get_random_forest_acc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



nn_hist, nn_acc, nn_conf = get_nn_acc()
cnn_hist, cnn_acc, cnn_conf = get_cnn_acc()
rf_acc, rf_conf = get_random_forest_acc()

# Print accuracies for each model
print("NN Accuracy:",  str(nn_acc))
print("CNN Accuracy:",  str(cnn_acc))
print("Random Forest Accuracy:", str(rf_acc))




# plot the nn accuracy vs cnn accuracy per 

# Plotting
plt.figure(figsize=(10, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(cnn_hist.history['val_loss'], label='CNN Val Loss')
plt.plot(nn_hist.history['val_loss'], label='NN Val Loss')
plt.title('Model Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(cnn_hist.history['val_accuracy'], label='CNN Val Accuracy')
plt.plot(nn_hist.history['val_accuracy'], label='NN Val Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# confusion matrix
def plot_confusion_matrix_seaborn(cm, class_labels, model):
    plt.figure(figsize=(15, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(model + ' Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

labels = ['Elec.',
    'Pop',
    'Expe.',
    'Indu.',
    'World',
    'Latin',
    'Hip-hop',
    'Rap',
    'Ehouse',
    'Folk',
    'Rock',
    'Reggae',
    'Lo-fi',
    'Instr',
    'ST']
plot_confusion_matrix_seaborn(rf_conf, labels, 'RF')
plot_confusion_matrix_seaborn(cnn_conf, labels, 'CNN')
plot_confusion_matrix_seaborn(nn_conf, labels, 'NN')

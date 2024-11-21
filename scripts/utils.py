import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_history(history, history_fine=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    plt.show()

    if history_fine:
        acc_fine = history_fine.history['accuracy']
        val_acc_fine = history_fine.history['val_accuracy']
        loss_fine = history_fine.history['loss']
        val_loss_fine = history_fine.history['val_loss']
        epochs_fine = range(len(acc_fine))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_fine, acc_fine, label='Fine-Tuning Training Accuracy')
        plt.plot(epochs_fine, val_acc_fine, label='Fine-Tuning Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Fine-Tuning Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_fine, loss_fine, label='Fine-Tuning Training Loss')
        plt.plot(epochs_fine, val_loss_fine, label='Fine-Tuning Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Fine-Tuning Training and Validation Loss')
        plt.savefig('fine_tuning_history.png')
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()

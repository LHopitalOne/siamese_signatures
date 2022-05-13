import tensorflow as tf
import matplotlib.pyplot as plt

def plot_history(hist):
    plt.figure()
   
    plt.plot(hist.history["binary_accuracy"])
    plt.plot(hist.history['val_binary_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(["Binary Accuracy", "Validation Binary Accuracy", "loss", "Validation Loss"])
    plt.xlabel("Epoch")
    
    plt.show()


    
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from models.siamese_model import siamese

from utils.data_generator import SiameseDataGenerator
from utils.misc import plot_history

from configs.parameters import properties

if __name__ == "__main__":
    train_dataset = SiameseDataGenerator(None, "./data", properties)

    model = siamese

    checkpoint_callback = ModelCheckpoint(
        filepath=properties["ckpt_path"],
        save_best_only=True,
        mode="min"
    )

    model.compile(
        loss="binary_crossentropy", 
        optimizer=Adam(learning_rate=properties["learning_rate"]), 
        metrics=["binary_accuracy"]
    )

    history = model.fit(
        train_dataset,
        epochs=properties["epochs"],
        callbacks=[checkpoint_callback]
    )

    plot_history(hist=history)

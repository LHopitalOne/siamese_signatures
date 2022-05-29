import os
from cv2 import TermCriteria_COUNT
from numpy.lib.histograms import histogram
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from models.siamese_model import siamese

from utils.data_generator import SiameseDataGenerator
from utils.misc import plot_history

from configs.parameters import properties

if __name__ == "__main__":
    train_dataset = SiameseDataGenerator(None, "./train", properties)

    model = siamese

    # x = np.ones((320, 2, 256, 256, 1))
    # y = np.ones((320, 1))

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

    # print(np.array(train_dataset[0][0]).shape)
    # print(len(train_dataset))

    # train_1 = [train_dataset[i][0][0] for i in range(len(train_dataset))]
    # train_2 = [train_dataset[i][0][1] for i in range(len(train_dataset))]
    # train_3 = [train_dataset[i][1] for i in range(len(train_dataset))]

    # print(np.array(train_1).shape)
    # print(np.array(train_2).shape)
    # print(np.array(train_3).shape)

    history = model.fit_generator(
        generator=train_dataset,
        epochs=properties["epochs"],
        callbacks=[checkpoint_callback]
    )

    # history = model.fit(
    #     x=[x[:, 0], x[:, 1]],
    #     y=y,
    #     epochs=20
    # )

    # plot_history(hist=history)
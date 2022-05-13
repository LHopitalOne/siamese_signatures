import tensorflow as tf
import numpy as np
import itertools
import os
import random
import cv2
import glob

class SiameseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, augmentations, directory, properties):
        self.augmentations = augmentations
        self.directory = directory
        self.batch_size = properties["batch_size"]
        self.dim = properties["target_size"]
        
        self._list_folders = self.__get_folders_list(directory)
        self.pair_combinations = self.__get_all_combinations()

        print(self.pair_combinations[:10])
        
    def __len__(self):
        return int(len(self._list_folders) / self.batch_size)
    
    def __getitem__(self, index):
        is_same = np.random.choice([False, True], p=[0.8, 0.2])
        
        x_batch = np.ones((self.batch_size, 2, self.dim[0], self.dim[1], 1))
        y_batch = np.ones((self.batch_size, 1))
        
        for i in range(self.batch_size):
            img1 = self.__load_img(self.pair_combinations[i][0])
            img2 = self.__load_img(self.pair_combinations[i][1])
    
            x_batch[i][0] = img1
            x_batch[i][1] = img2

            if self.pair_combinations[i][0][len(self.directory) + 1 : len(self.directory) + 4] == self.pair_combinations[i][1][len(self.directory) + 1 : len(self.directory) + 4]:
                y_batch[i] = 1.0
            else:
                y_batch[i] = 0.0
                        
        return x_batch, y_batch
          
    def on_epoch_end(self):
        np.random.shuffle(self.pair_combinations)
        
    def __load_img(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, self.dim)
        img = np.expand_dims(img, axis=-1).astype(np.float32) / 255.0
        
        return img
    
    def __get_all_combinations(self):
        return sorted(list(itertools.combinations(glob.glob(os.path.join(self.directory, "*", "*")), 2)))
    
    def __get_folders_list(self, path):
        _list = glob.glob(os.path.join(self.directory, "*"))
        
        return _list

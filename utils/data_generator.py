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
        # self.pair_combinations = self.__get_all_combinations()

        print(self._list_folders[:10])
        
    def __len__(self):
        return int(len(self._list_folders) / self.batch_size)
    
    def __getitem__(self, index):
        x_batch = np.ones((self.batch_size, 2, self.dim[0], self.dim[1], 1))
        y_batch = np.ones((self.batch_size, 1))

        print("batch size:", self.batch_size)

        for i in range(self.batch_size):
            is_same = np.random.choice([0, 1], p=[0.8, 0.2])

            img1_path = np.random.choice(self._list_folders)
            img1 = self.__load_img(np.random.choice(glob.glob(os.path.join(self.directory, img1_path) + "/*.*")))
            
            # print("is_same:", is_same)

            if is_same == 1:
                img2 = self.__load_img(np.random.choice(glob.glob(os.path.join(self.directory, img1_path) + "/*.*")))

                # print("true y_batch:", y_batch, "i_true ", i)
            else:
                img2_path = np.random.choice(self._list_folders)
                while img2_path == img1_path:
                    img2_path = np.random.choice(self._list_folders)
                img2 = self.__load_img(np.random.choice(glob.glob(os.path.join(self.directory, img2_path) + "/*.*")))

                y_batch[i] = 0
                
                # print("false batch:", y_batch, "i_false: ", i)

            x_batch[i][0] = img1
            x_batch[i][1] = img2

        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self._list_folders)
        
    def __load_img(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, self.dim)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1).astype(np.float32) / 255.0

        # print("shape:", img.shape)
        
        return img
    
    def __get_all_combinations(self):
        return sorted(list(itertools.combinations(glob.glob(os.path.join(self.directory, "*", "*")), 2)))
    
    def __get_folders_list(self, path):
        _list = glob.glob(os.path.join(path, "*"))

        for i in range(len(_list)):
            _list[i] = _list[i][len(path) + 1:]
        
        return _list

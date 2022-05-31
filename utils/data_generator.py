import itertools
import os
import cv2
import glob
import tensorflow as tf
import numpy as np

class SiameseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, augmentations, properties):
        self.augmentations = augmentations
        self.directory = directory
        self.batch_size = properties["batch_size"]
        self.dim = properties["target_size"]

        self._list_folders = self.__get_folders_list(directory)

        print(self._list_folders[:10])

    def __len__(self):
        return int(len(self._list_folders) / self.batch_size)

    def __getitem__(self, index):
        x1_batch = np.ones((self.batch_size, self.dim[0], self.dim[1], 1))
        x2_batch = np.ones((self.batch_size, self.dim[0], self.dim[1], 1))
        
        y_batch = np.ones((self.batch_size, 1))

        _list_folders = self._list_folders[index * self.batch_size: (index + 1) * self.batch_size]

        for i in range(self.batch_size):
            is_same = np.random.choice([0, 1], p=[0.8, 0.2])

            img1_path = np.random.choice(_list_folders)
            img1_folder = img1_path[len(self.directory) + 1:len(self.directory) + 4]

            img1 = self.__load_img(img1_path)
            # img1 = self.__load_img(np.random.choice(glob.glob(os.path.join(self.directory, img1_path) + "/*.*")))

            # print("is_same:", is_same)

            if is_same == 1:
                img2_path = np.random.choice(glob.glob(os.path.join(self.directory, img1_folder,  "*.*")))
                img2 = self.__load_img(img2_path)

                print("true img1/img2:", img1_path, img2_path)
                # img2 = self.__load_img(np.random.choice(glob.glob(os.path.join(self.directory, img1_path) + "/*.*")))
            else:
                img2_path = np.random.choice(_list_folders)
                img2_folder = img2_path[len(self.directory) + 1:len(self.directory) + 4]

                while img2_folder == img1_folder:
                    img2_path = np.random.choice(_list_folders)
                    img2_folder = img2_path[len(self.directory) + 1:len(self.directory) + 4]
                
                img2 = self.__load_img(img2_path)
                # img2 = self.__load_img(np.random.choice(glob.glob(os.path.join(self.directory, img2_path) + "/*.*")))

                y_batch[i] = 0

                # print("false img1/img2:", img1_path, img2_path)
                # print("false batch:", y_batch, "i_false: ", i)

            x1_batch[i] = img1
            x2_batch[i] = img2

        return (x1_batch, x2_batch), y_batch

    def on_epoch_end(self):
        np.random.shuffle(self._list_folders)

    def __load_img(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, self.dim)
        # img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1).astype(np.float32) / 255.0

        # print("shape:", img.shape)

        return img

    def __get_folders_list(self, path):
        # TODO: this function should read all images from each folder
        #       the rest of the code should be modified accordingly
        _list = glob.glob(os.path.join(path, "*/*"))

        # for i in range(len(_list)):
        #     _list[i] = _list[i][len(path) + 1:]

        return _list

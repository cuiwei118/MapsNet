import os
import random
import numpy as np
import keras
from libtiff import TIFF
import glob
import cv2
from PIL import Image


def adjustData(img, mask):
    img = img / 255.0
    mask[mask >= 1] = 1
    mask[mask < 1] = 0
    return (img, mask)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_path, images1folder,images2folder, mask_folder, batch_size=2,
                 target_size=(512, 512), shuffle=True):
        im1_fns = glob.glob(os.path.join(train_path, images1folder, '*.jpg')) #返回符合匹配所有文件的路径
        self.data = []
        for im1_fn in im1_fns:
            im2_fn = os.path.join(train_path, images2folder, os.path.basename(im1_fn).replace('.jpg', '.jpg'))
            mask_fn = os.path.join(train_path, mask_folder, os.path.basename(im1_fn).replace('.jpg', '.png'))

            self.data.append([im1_fn,im2_fn,mask_fn])
        self.list_IDs = list(range(len(self.data)))
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        [x1,x2], y = self.__data_generation(list_IDs_temp)

        return [x1,x2], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        imgs1 = []
        imgs2 = []
        masks = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            im1_fn, im2_fn, mask_fn = self.data[ID]
            img1 = cv2.imread(im1_fn)
            img2 = cv2.imread(im2_fn)
            mask = cv2.imread(mask_fn, 0)
            mask = np.expand_dims(mask, -1)

            img1, mask = adjustData(img1, mask)
            img2, mask = adjustData(img2, mask)

            imgs1.append(img1)
            imgs2.append(img2)
            masks.append(mask)

        return [np.array(imgs1), np.array(imgs2)], np.array(masks)




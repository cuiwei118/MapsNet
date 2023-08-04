import os
import argparse
import numpy as np
from model.siameseB import MCDU
from DSFIN import DSIFN
from model.FC_EF import FC_EF
from model.FC_Siam_differ import FC_Siam_differ
from PIL import Image
from fcsc import get_FCSC_model
import cv2
from tqdm import tqdm
from libtiff import TIFF
import glob

def parse_args():
    parser = argparse.ArgumentParser('unet segmentation')
    parser.add_argument('-i', '--input1', type=str, default='../DSFIN/test/t1', help='input1 filename or directory')
    parser.add_argument('-d', '--input2', type=str, default='../DSFIN/test/t2', help='input2 filename or directory')
    parser.add_argument('-s', '--size', type=int, default=512, help='input size')
    parser.add_argument('-c', '--checkpoint', type=str,
        default='../model/checkpoints/my_best_model048.h5', help='input segmentation model')
    parser.add_argument('-o', '--output', type=str, default='results', help='output directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    SIZE = args.size
    target_size = (SIZE, SIZE)
    model = MCDU()
    model.load_weights(args.checkpoint)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if os.path.isfile(args.input1)and os.path.isfile(args.input2):

        img1 = cv2.imread(args.input1)
        img2 = cv2.imread(args.input2)
        original_size = img1.shape[0:2]
        original_size = img2.shape[0:2]
        img1 = cv2.resize(img1, target_size)
        img2 = cv2.resize(img2, target_size)
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        img1 = np.reshape(img1, (1,) + img1.shape)
        img2 = np.reshape(img2, (1,) + img2.shape)
        result = model.predict(img1,img2)[0]
        result[result >= 0.5] = 255
        result[result < 0.5] = 0
        result = cv2.resize(result, original_size)
        im_fn = os.path.basename(args.input)
        cv2.imwrite(os.path.join(args.output, os.path.splitext(im_fn)[0] + '.png'), result)
    else:
        im1_fns = os.listdir(args.input1)    # 这个地址下的所有文件目录
        im2_fns = os.listdir(args.input2)
        im1_fns.sort()
        im2_fns.sort()
        for im_fn in tqdm(im1_fns):
            img1 = cv2.imread(os.path.join(args.input1,im_fn))
            img2 =cv2.imread(os.path.join(args.input2,im_fn))

            img1 = img1 / 255.0
            img2 = img2 / 255.0
            img1 = np.reshape(img1, (1,) + img1.shape)
            img2 = np.reshape(img2, (1,) +img2.shape)

            result = model.predict([img1,img2])[0]
            result[result >= 0.5] = 255
            result[result < 0.5] = 0
            cv2.imwrite(os.path.join(args.output, os.path.splitext(im_fn)[0] + '.png'), result)

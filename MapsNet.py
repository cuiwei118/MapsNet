import os
from keras import *
from keras.models import Model
from keras.layers import *
from keras import applications
from keras import backend as K
from keras.utils import conv_utils
import cv2 as cv
from attention import cbam_block
IMAGE_ORDERING = 'channels_last'
from ASPP import ASPP
import tensorflow  as tf
import numpy as np


def MapsNet():
    vgg_model = applications.VGG16(weights='imagenet',include_top=False,input_shape=(512,512,3))
    b5c3_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block5_conv3').output) # 调用vgg第五层
    b5c3_model.trainable=True

    b4c3_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block4_conv3').output) # 调用vgg第四层
    b4c3_model.trainable=False

    b3c3_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block3_conv3').output) # 调用vgg第三层
    b3c3_model.trainable=False

    b2c2_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block2_conv2').output) # 调用vgg第二层
    b2c2_model.trainable=False

    b1c2_model = Model(inputs=vgg_model.input, outputs = vgg_model.get_layer('block1_conv2').output) # 调用vgg第一层
    b1c2_model.trainable=False

    input_t1 = layers.Input((512,512,3), name='Input_t1')
    input_t2 = layers.Input((512,512,3), name='Input_t2')

    t1_b5c3 = b5c3_model(input_t1) #512
    t1_b5c3 = cbam_block(t1_b5c3,ratio=8)
    t2_b5c3 = b5c3_model(input_t2)
    t2_b5c3 = cbam_block(t2_b5c3,ratio=8)
    concat5 = subtract([t1_b5c3,t2_b5c3])

    t1_b4c3 = b4c3_model(input_t1)  #512
    t1_b4c3 = cbam_block(t1_b4c3,ratio=8)
    t2_b4c3 = b4c3_model(input_t2)
    t2_b4c3 = cbam_block(t2_b4c3,ratio=8)
    concat4 = subtract([t1_b4c3, t2_b4c3])

    t1_b3c3 = b3c3_model(input_t1)
    t1_b3c3 = cbam_block(t1_b3c3,ratio=8)
    t2_b3c3 = b3c3_model(input_t2)  #256
    t2_b3c3 = cbam_block(t2_b3c3,ratio=8)
    concat3 = subtract([t1_b3c3, t2_b3c3])

    t1_b2c3 = b2c2_model(input_t1)  #128
    t1_b2c3 = cbam_block(t1_b2c3,ratio=8)
    t2_b2c3 = b2c2_model(input_t2)
    t2_b2c3 = cbam_block(t2_b2c3,ratio=8)
    concat2 = subtract([t1_b2c3, t2_b2c3])

    t1_b1c3 = b1c2_model(input_t1)  #64
    t1_b1c3 = cbam_block(t1_b1c3,ratio=8)
    t2_b1c3 = b1c2_model(input_t2)
    t2_b1c3 = cbam_block(t2_b1c3,ratio=8)
    concat1 = subtract([t1_b1c3, t2_b1c3])
    X = ASPP(concat5)
    X_1 = tf.nn.depth_to_space(X,16)
    pxielsuffle0 = Conv2D(1, (1, 1), activation='sigmoid',padding='same')(X_1)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(X), concat4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6_1 = tf.nn.depth_to_space(conv6,8)
    pxielsuffle1 = Conv2D(1, (1, 1), activation='sigmoid',padding='same')(conv6_1)
    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), concat3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Dropout(0.5)(conv7)
    conv7_1 = tf.nn.depth_to_space(conv7,4)
    pxielsuffle2 = Conv2D(1, (1, 1), activation='sigmoid',padding='same')(conv7_1)
    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), concat2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Dropout(0.5)(conv8)
    conv8_1 = tf.nn.depth_to_space(conv8,2)
    pxielsuffle3 = Conv2D(1, (1, 1), activation='sigmoid',padding='same')(conv8_1)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), concat1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Dropout(0.5)(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid',name='out2')(conv9)
    conv10 = concatenate([conv10,pxielsuffle0,pxielsuffle1,pxielsuffle2,pxielsuffle3],axis =3)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid',name='out1')(conv10)
    output = Model(inputs=[input_t1,input_t2], outputs=conv10)

    return output








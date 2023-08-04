from keras import *
from keras.layers import *
import os

def CDNet():
    input_1 = Input(shape =(256, 256, 3), name='Input_1')
    input_2 = Input(shape =(256, 256, 3), name='Input_2')

    input = concatenate([input_1,input_2],axis=3)
    x = Conv2D(64,(7,7),strides=(1,1),activation='relu',padding='same')(input)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)
    x = Conv2D(64,(7,7),strides=(1,1),activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)
    x = Conv2D(64,(7,7),strides=(1,1),activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)
    x = Conv2D(64,(7,7),strides=(1,1),activation='relu',padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)

    x = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = Conv2D(64, (7, 7), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = Conv2D(64, (7, 7), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = Conv2D(64, (7, 7), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = UpSampling2D((2,2),interpolation='bilinear')(x)
    x = Conv2D(64, (7, 7), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(1,(1,1),activation='sigmoid')(x
    CDNet = Model(inputs=[input_1,input_2], outputs=x)
    return CDNet





from keras import *
from keras.layers import *
import os

def FC_EF():
    input_1 = Input(shape =(512, 512, 3), name='Input_1')
    input_2 = Input(shape =(512, 512, 3), name='Input_2')
    input = concatenate([input_1,input_2],axis=3)
    m = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = BatchNormalization(axis=3)(m)
    x = Dropout(0.5)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x1 = Dropout(0.5, name='block1_drop')(x)   # 节点1
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x2 = Dropout(0.5, name='block2_drop')(x)   # 节点2
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x3 = Dropout(0.5, name='block3_drop')(x)  # 节点3
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x4 = Dropout(0.5, name='block4_drop')(x)  # 节点4

    x = MaxPooling2D((2, 2), strides=(2, 2))(x4)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same' )(x)  #上采样
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = concatenate([x,x4],axis=3)

    x = Conv2DTranspose(128,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(128,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(64,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = concatenate([x,x3],axis=3)

    x = Conv2DTranspose(64,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(64,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(32,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = concatenate([x,x2],axis=3)

    x = Conv2DTranspose(32,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(16,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = concatenate([x,x1],axis=3)

    x = Conv2DTranspose(16,kernel_size=(3,3), padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2DTranspose(1,kernel_size=(3,3), padding='same',activation='sigmoid')(x)
    FC_EF = Model(inputs=[input_1,input_2], outputs=x)
    return FC_EF

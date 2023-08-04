from keras import *
from keras.layers import *
import os
from keras import backend as K
from keras.utils.vis_utils import plot_model
os.environ["PATH"] += os.pathsep + 'D:/其他工具/Graphviz/bin'

IMAGE_ORDERING = 'channels_last'
def ppm(input):

    x = input
    shapex = K.int_shape(x)[3]
    p1 = MaxPool2D((1,1),strides=(1,1),name='ppm_pool1')(x)
    p2 = MaxPool2D((2, 2), strides=(2, 2), name='ppm_pool2')(x)
    p3 = MaxPool2D((4, 4), strides=(4, 4), name='ppm_pool3')(x)


    c1 = Conv2D(shapex,(10,10),activation='relu',padding='same',name='ppm_conv1')(p1)
    c2 = Conv2D(shapex,(15,15),activation='relu',padding='same',name='ppm_conv2')(p2)
    c3 = Conv2D(shapex,(20,20),activation='relu',padding='same',name='ppm_conv3')(p3)



    u1 = UpSampling2D((1, 1), data_format=IMAGE_ORDERING)(c1)
    u2 = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(c2)
    u3 = UpSampling2D((4, 4), data_format=IMAGE_ORDERING)(c3)
    # u4 = UpSampling2D((8, 8), data_format=IMAGE_ORDERING)(c4)

    j = concatenate([input,u1,u2,u3],axis=3)
    out = Conv2D(shapex,(1,1),strides=(1,1),padding='same',data_format=IMAGE_ORDERING)(j)

    return out

def FCN_PP():
    input_1 = Input(shape =(512, 512, 3), name='Input_1')
    input_2 = Input(shape =(512, 512, 3), name='Input_2')
    input = concatenate([input_1,input_2],axis=3)
    m = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = BatchNormalization(axis=3)(m)
    x = Dropout(0.5)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)   # 节点1
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5, name='block1_drop')(x)
    x1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x1)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5, name='block2_drop')(x)   # 节点2
    x2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5, name='block3_drop')(x)  # 节点3
    x3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5, name='block4_drop')(x)  # 节点4
    x = ppm(x)
    x = concatenate([x,x3],axis=3)

    x = UpSampling2D((2, 2),interpolation='bilinear')(x)
    x = concatenate([x, x2],axis=3)
    x = Conv2D(64, (3, 3),activation='relu', padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)

    x = UpSampling2D((2, 2),interpolation='bilinear')(x)
    x = concatenate([x, x1],axis=3)
    x = Conv2D(32, (3, 3), activation='relu', padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same' )(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)

    x = UpSampling2D((2, 2),interpolation='bilinear')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1,kernel_size=(3,3), padding='same',activation='sigmoid')(x)

    FCN_PP = Model(inputs=[input_1,input_2], outputs=x)
    plot_model(FCN_PP, to_file='model3.png')
    return FCN_PP

FCN_PP().summary()
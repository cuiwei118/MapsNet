import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Lambda, Subtract,BatchNormalization,\
    Conv2DTranspose
from keras.models import Input, Model
from keras.utils.vis_utils import plot_model
import os


def get_FCSC_model(input_size=(512,512,3), pre_weights=None):
    # get a Siamese Encoder
    inputs_tensor = Input(shape=input_size)
    Contract_Path_Model = Model(inputs=[inputs_tensor], outputs=contract_path(inputs_tensor))
    Inputs_1 = Input(shape=input_size)
    Inputs_2 = Input(shape=input_size)
    _, feature_11, feature_12, feature_13, feature_14 = Contract_Path_Model(Inputs_1)
    feature_2, feature_21, feature_22, feature_23, feature_24 = Contract_Path_Model(Inputs_2)

    # get a Decoder
    FSCD_model = Model(inputs=[Inputs_1, Inputs_2],
                       outputs=expansive_path(feature_2, feature_11, feature_21, feature_12, feature_22, feature_13,
                                              feature_23, feature_14, feature_24))

    return FSCD_model


def Abs_layer(tensor):
    return Lambda(K.abs)(tensor)


def contract_path(Inputs):
    Conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Inputs)
    Conv_1 = BatchNormalization()(Conv_1)
    Conv_1 = Dropout(0.2)(Conv_1)
    Conv_1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_1)
    Conv_1 = BatchNormalization()(Conv_1)
    Conv_1 = Dropout(0.2)(Conv_1)

    feature_1 = Conv_1
    Pool_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_1)

    Conv_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_1)
    Conv_2 = BatchNormalization()(Conv_2)
    Conv_2 = Dropout(0.2)(Conv_2)
    Conv_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_2)
    Conv_2 = BatchNormalization()(Conv_2)
    Conv_2 = Dropout(0.2)(Conv_2)

    feature_2 = Conv_2
    Pool_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_2)

    Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_2)
    Conv_3 = BatchNormalization()(Conv_3)
    Conv_3 = Dropout(0.2)(Conv_3)
    Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_3)
    Conv_3 = BatchNormalization()(Conv_3)
    Conv_3 = Dropout(0.2)(Conv_3)
    Conv_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_3)
    Conv_3 = BatchNormalization()(Conv_3)
    Conv_3 = Dropout(0.2)(Conv_3)

    feature_3 = Conv_3
    Pool_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_3)

    Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Pool_3)
    Conv_4 = BatchNormalization()(Conv_4)
    Conv_4 = Dropout(0.2)(Conv_4)
    Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_4)
    Conv_4 = BatchNormalization()(Conv_4)
    Conv_4 = Dropout(0.2)(Conv_4)
    Conv_4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Conv_4)
    Conv_4 = BatchNormalization()(Conv_4)
    Conv_4 = Dropout(0.2)(Conv_4)

    feature_4 = Conv_4
    Pool_4 = MaxPooling2D(pool_size=(2, 2), padding='same')(Conv_4)


    return Pool_4, feature_1, feature_2, feature_3, feature_4


def expansive_path(feature, fea_11, fea_21, fea_12, fea_22, fea_13, fea_23, fea_14, fea_24):
    Up_1 = Conv2DTranspose(128, 3,strides=(2, 2 ) ,activation='relu', padding='same', kernel_initializer='he_normal')(feature)
    Merge_1 = Concatenate()([fea_14, fea_24, Up_1])
    Deconv_1 = Conv2DTranspose(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_1)
    Deconv_1 = BatchNormalization()(Deconv_1)
    Deconv_1 = Dropout(0.2)(Deconv_1)
    Deconv_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_1)
    Deconv_1 = BatchNormalization()(Deconv_1)
    Deconv_1 = Dropout(0.2)(Deconv_1)
    Deconv_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_1)
    Deconv_1 = BatchNormalization()(Deconv_1)
    Deconv_1 = Dropout(0.2)(Deconv_1)

    Up_2 = Conv2DTranspose(64, 3,strides=(2, 2 ) ,activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_1)
    Up_2 = BatchNormalization()(Up_2)
    Up_2 = Dropout(0.2)(Up_2)
    Merge_2 = Concatenate(axis=-1)([fea_13, fea_23, Up_2])


    Deconv_2 = Conv2DTranspose(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_2)
    Deconv_2 = BatchNormalization()(Deconv_2)
    Deconv_2 = Dropout(0.2)(Deconv_2)
    Deconv_2 = Conv2DTranspose(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_2)
    Deconv_2 = BatchNormalization()(Deconv_2)
    Deconv_2 = Dropout(0.2)(Deconv_2)
    Deconv_2 = Conv2DTranspose(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_2)
    Deconv_2 = BatchNormalization()(Deconv_2)
    Deconv_2 = Dropout(0.2)(Deconv_2)

    Up_3 = Conv2DTranspose(32, 3,strides=(2,2), activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_2)
    Up_3 = BatchNormalization()(Up_3)
    Up_3 = Dropout(0.2)(Up_3)
    Merge_3 = Concatenate(axis=-1)([fea_12, fea_22, Up_3])


    Deconv_3 = Conv2DTranspose(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_3)
    Deconv_3 = BatchNormalization()(Deconv_3)
    Deconv_3 = Dropout(0.2)(Deconv_3)
    Deconv_3 = Conv2DTranspose(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_3)
    Deconv_3 = BatchNormalization()(Deconv_3)
    Deconv_3 = Dropout(0.2)(Deconv_3)

    Up_4 = Conv2DTranspose(16, 2, strides=(2,2),activation='relu', padding='same', kernel_initializer='he_normal')(Deconv_3)
    Up_4 = BatchNormalization()(Up_4)
    Up_4 = Dropout(0.2)(Up_4)
    Merge_4 = Concatenate(axis=-1)([fea_11, fea_21, Up_4])


    Deconv_4 = Conv2DTranspose(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(Merge_4)
    Deconv_4 = BatchNormalization()(Deconv_4)
    Deconv_4 = Dropout(0.2)(Deconv_4)

    logits   =  Conv2DTranspose(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(Deconv_4)

    return logits


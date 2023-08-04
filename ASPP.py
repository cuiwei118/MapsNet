from tensorflow import keras
import numpy as np
def ASPP(inputs, depthwise=False, output_stride=16):
    """
    Following Code from Here:
    https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0/blob/master/deeplab.py
    For the ASPP Module, will end up having two options of with depthwise and without.
    """
    dilation_rates = np.array([(6, 6), (12, 12), (18, 18)])
    if output_stride == 8:
        dilation_rates = dilation_rates*2
    shape = list(inputs.shape)

    # Image pooling
    pool = keras.layers.AveragePooling2D(pool_size=(
        shape[1], shape[2]), name="ASPP_Ave_Pool")(inputs)
    conv1 = keras.layers.Conv2D(
        256, 1, strides=1, padding='same', use_bias=False, name='ASPP_conv1')(pool)
    norm1 = keras.layers.BatchNormalization(
        name='ASPP_conv1_batch_norm')(conv1)
    relu1 = keras.layers.Activation('relu', name='ASPP_conv1_relu')(norm1)
    upsampling = keras.layers.UpSampling2D(
        size=(shape[1], shape[2]), interpolation='bilinear')(relu1)

    # 1x1 Convolution
    conv1x1 = keras.layers.Conv2D(
        256, 1, strides=1, padding='same', use_bias=False, name='ASPP_conv1x1')(inputs)
    norm1x1 = keras.layers.BatchNormalization(
        name='ASPP_conv1x1_batch_norm')(conv1x1)
    relu1x1 = keras.layers.Activation(
        'relu', name='ASPP_conv1x1_relu')(norm1x1)

    # The Dilated Convolutions
    if depthwise:
        conv3x3_d6 = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[0], name="ASPP_sep_conv3x3_d6")(inputs)
    else:
        conv3x3_d6 = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[0], name="ASPP_conv3x3_d6")(inputs)
    norm3x3_d6 = keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d6_batch_norm")(conv3x3_d6)
    relu3x3_d6 = keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d6_relu")(norm3x3_d6)
    if depthwise:
        conv3x3_d12 = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[1], name="ASPP_sep_conv3x3_d12")(inputs)
    else:
        conv3x3_d12 = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[1], name="ASPP_conv3x3_d12")(inputs)
    norm3x3_d12 = keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d12_batch_norm")(conv3x3_d12)
    relu3x3_d12 = keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d12_relu")(norm3x3_d12)
    if depthwise:
        conv3x3_d18 = keras.layers.SeparableConv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[2], name="ASPP_sep_conv3x3_d18")(inputs)
    else:
        conv3x3_d18 = keras.layers.Conv2D(
            256, 3, strides=1, padding='same', dilation_rate=dilation_rates[2], name="ASPP_conv3x3_d18")(inputs)
    norm3x3_d18 = keras.layers.BatchNormalization(
        name="ASPP_Sep_conv3x3_d18_batch_norm")(conv3x3_d18)
    relu3x3_d18 = keras.layers.Activation(
        'relu', name="ASPP_Sep_conv3x3_d18_relu")(norm3x3_d18)

    # Concatenate all the above layers
    concat = keras.layers.Concatenate(name='ASPP_concatenate')(
        [upsampling, relu1x1, relu3x3_d6, relu3x3_d12, relu3x3_d18])

    # Do the final convolution
    conv2 = keras.layers.Conv2D(
        256, 1, strides=1, padding='same', use_bias=False, name='ASPP_project_conv')(concat)
    norm2 = keras.layers.BatchNormalization(
        name='ASPP_project_conv_batch_norm')(conv2)
    relu2 = keras.layers.Activation(
        'relu', name='ASPP_project_conv_relu')(norm2)

    # Return the result
    return relu2
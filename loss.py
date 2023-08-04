from tensorflow.keras import backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def dice_coef(y_true, y_pred, smooth = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def accuracy(y_true, y_pred):

    return K.mean(K.equal(y_true, K.round(y_pred)))

def bce_dice_loss(y_true,y_pred):
    sig_y_true = K.sigmoid(y_true)
    sig_y_pred = K.sigmoid(y_pred)

    dice_loss = 1 - dice_coef(y_true,y_pred)

    return binary_crossentropy(sig_y_true,sig_y_pred) + dice_loss
#
# def dice_focal_loss(y_true,y_pred):

def F1(y_true, y_pred):
    Precision = precision(y_true,y_pred)
    Recall = recall(y_true,y_pred)
    f1 = 2 *((Precision * Recall) / (Precision + Recall +K.epsilon()))
    return f1

def weighted_bce_dice_loss(y_true,y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])

    class_weights = [0.1, 0.9]#note that the weights can be computed automatically using the training smaples
    weighted_bce = K.sum(class_loglosses * K.constant(class_weights))
    dice_coef_loss = 1 - dice_coef(y_true,y_pred)
    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work
    return  weighted_bce + 0.7 * dice_coef_loss


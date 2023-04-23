import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        iou = (intersection + 1e-15) / (union + 1e-15)
        iou = iou.astype(np.float32)
        return iou
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

epsilon = 1e-15
def dice_coef(y_true, y_pred):
    y_true_flat = tf.keras.layers.Flatten()(y_true)
    y_pred_flat = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice = (2. * intersection + epsilon) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + epsilon)
    return dice

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

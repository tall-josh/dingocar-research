from tensorflow.python import keras
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.keras.layers import Input, Dense
import tensorflow.python.keras.backend as K

def is_sim_linear_loss(y_true, y_pred):
    gt_angle    = y_true[0]
    gt_throttle = y_true[1]
    gt_is_sim   = y_true[2]
    point_5     = y_true[3]

    pred_angle    = y_pred[0]
    pred_throttle = y_pred[1]
    pred_is_sim   = y_pred[2]
    pred_smoosh   = y_pred[3]

    # Multiply by gt_is_sim so we only use loss from the simulated data.
    # If gt_is_sim is zero this will drive the loss to zero.
    loss  = K.mean(K.square(pred_angle-gt_angle))       * gt_is_sim
    loss += K.mean(K.square(pred_throttle-gt_throttle)) * gt_is_sim
    loss += K.mean(K.square(pred_is_sim-gt_is_sim))
    loss += K.mean(K.square(pred_smoosh - point_5))

    return loss

def is_sim_categorical_loss(y_true, y_pred):
    gt_angle    = y_true[0]
    gt_throttle = y_true[1]
    gt_is_sim   = y_true[2]
    point_5     = y_true[3]

    pred_angle    = y_pred[0]
    pred_throttle = y_pred[1]
    pred_is_sim   = y_pred[2]
    pred_smoosh   = y_pred[3]

    loss  = K.mean(K.square(pred_angle-gt_angle))
    loss += K.mean(K.square(pred_throttle-gt_throttle))

    #categorical_crossentropy
    is_sim_loss = K.mean(K.categorical_crossentropy(pred_is_sim, gt_is_sim))
    loss += K.mean(K.categorical_crossentropy(pred_smoosh, point_5))
    return loss

def control_only_metric(y_true, y_pred):
    gt_angle      = y_true[0]
    gt_throttle   = y_true[1]

    pred_angle    = y_pred[0]
    pred_throttle = y_pred[1]

    return K.mean(K.square(pred_angle-gt_angle)) +\
           K.mean(K.square(pred_throttle-gt_throttle))


from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense
import tensorflow.python.keras.backend as K

def is_sim_linear_loss():
    loss = {"steering_output" : "mse",
            "throttle_output" : "mse",
            "is_sim_output"   : "mse",
            "smoosh_output"   : "mse",
            }

    return loss

def is_sim_categorical_loss():
    loss = {"steering_output" : "mse",
            "throttle_output" : "mse",
            "is_sim_output"   : "categorical_crossentropy",
            "smoosh_output"   : "categorical_crossentropy",
            }
    return loss

def control_only_metric(y_true, y_pred):
    gt_angle      = y_true[0]
    gt_throttle   = y_true[1]

    pred_angle    = y_pred[0]
    pred_throttle = y_pred[1]

    return K.mean(K.square(pred_angle-gt_angle)) +\
           K.mean(K.square(pred_throttle-gt_throttle))


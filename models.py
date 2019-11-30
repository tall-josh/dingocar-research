import os
import numpy as np
from glob import glob

from tensorflow import ConfigProto, Session
from tensorflow.python import keras
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.keras.layers import Input, Dense
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.merge import concatenate

from generators import get_gens
from layers import default_n_linear, smoosh_classification, smoosh_linear

__all__ = ["KerasLinear", "CrossentropySmoosh", "LinearSmoosh"]

class KerasPilot(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car.
    '''
    def __init__(self, model):
        self.model = model
        self.optimizer = "adam"

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        assert False, "Not implemented :-("

    def compile(self):
        assert False, "Not implemented :-("

    def train(self, train_gen, val_gen, train_steps, val_steps,
              saved_model_path, epochs=100,
              verbose=1, min_delta=.0005, patience=5, monitor="val_loss",
              early_stop_callback = None, save_best_callback=None,
              other_callbacks = []):

        """
        train_gen: generator that yields an array of images an array of

        """

        #checkpoint to save model after each epoch
        if save_best_callback is not None:
            save_best = save_best_callback
        else:
            save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                        monitor=monitor,
                                                        verbose=verbose,
                                                        save_best_only=True,
                                                        mode='min')

        #stop training if the validation error stops improving.
        if early_stop_callback is not None:
            early_stop = early_stop_callback
        else:
            early_stop = keras.callbacks.EarlyStopping(monitor=monitor,
                                                   min_delta=min_delta,
                                                   patience=patience,
                                                   verbose=verbose,
                                                   mode='min')

        callbacks_list = [save_best, early_stop] + other_callbacks

        hist = self.model.fit_generator(
                        train_gen,
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        callbacks=callbacks_list,
                        validation_steps=val_steps)
        return hist

    def run(self, img_arr):
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return float(steering[0][0]), float(throttle[0][0])

    def get_features(self, img_arr):
        lp = 0 # learning_phase
        inp = self.model.input
        out = self.model.layers[14].output
        func = K.function([inp, K.learning_phase()], out)
        features = func([img_arr, lp])
        return features[0].tolist()



################################################################################

class KerasLinear(KerasPilot):
    '''
    The KerasLinear pilot uses one neuron to output a continous value via the 
    Keras Dense layer with linear activation. One each for steering and throttle.
    The output is not bounded.
    '''
    def __init__(self):
        super(KerasLinear, self).__init__(default_n_linear())

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                loss='mse')

################################################################################

def loss(y_true, y_pred):
    gt_angle    = y_true[0]
    gt_throttle = y_true[1]
    gt_is_sim   = y_true[2]
    point_5     = y_true[3]

    pred_angle    = y_pred[0]
    pred_throttle = y_pred[1]
    pred_is_sim   = y_pred[2]
    pred_smoosh   = y_pred[3]

    loss        = K.mean(K.square(pred_angle-gt_angle))
    loss       += K.mean(K.square(pred_throttle-gt_throttle))

    # Multiply by gt_is_sim so we only use loss from the simulated data.
    # If gt_is_sim is zero this will drive the loss to zero.
    is_sim_loss = K.mean(K.square(pred_is_sim-gt_is_sim)) * gt_is_sim
    smoosh_loss = K.mean(K.square(pred_smoosh - point_5))

    loss += is_sim_loss
    loss += smoosh_loss
    return loss

def control_only_metric(y_true, y_pred):
    gt_angle      = y_true[0]
    gt_throttle   = y_true[1]

    pred_angle    = y_pred[0]
    pred_throttle = y_pred[1]

    return K.mean(K.square(pred_angle-gt_angle)) +\
           K.mean(K.square(pred_throttle-gt_throttle))

class CopyWeights(keras.callbacks.Callback):

    def __init__(self, kl):
        self.kl = kl

    def on_train_batch_begin(self, batch, logs=None):
        kl = self.kl
        names = [l.name for l in kl.model.layers]
        # ORDER of result not gaurenteed. So we sort.
        layers = [l for l in kl.model.layers if l.name in ["is_sim", "smoosh"]]
        layers = sorted(layers, key = lambda x: x.name)
        is_sim, smoosh = layers
        assert "is_sim" in is_sim.name, f"'is_sim' != '{is_sim.name}'"
        assert "smoosh" in smoosh.name, f"'smoosh' != '{smoosh.name}'"

        smoosh.set_weights(is_sim.get_weights())

class LinearSmoosh(KerasPilot):
    '''
    The KerasLinear pilot uses one neuron to output a continous value via the
    Keras Dense layer with linear activation. One each for steering and throttle.
    The output is not bounded.
    '''
    def __init__(self):
        super(LinearSmoosh, self).__init__(smoosh_linear())

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=loss)#, metrics=[control_only_metric])
        print(f"Availiable metrics: {self.model.metrics_names}")


    def train(self, train_gen, val_gen, train_steps, val_steps,
              saved_model_path, epochs=100,
              verbose=1, min_delta=.0005, patience=5):

        copy_weights_callback = CopyWeights(self)
        return super().train(train_gen, val_gen, train_steps, val_steps,
                           saved_model_path, epochs=epochs,
                           verbose=verbose, min_delta=min_delta,
                           patience=patience,
                           monitor="val_steering_loss",
                           other_callbacks = [copy_weights_callback])

################################################################################

def loss(y_true, y_pred):
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

class CrossentropySmoosh(KerasPilot):
    '''

    '''
    def __init__(self):
        super(CrossentropySmoosh, self).__init__(smoosh_classification())

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=loss, metrics={"control_only_metric" : control_only_metric})


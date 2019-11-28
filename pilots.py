import os
import numpy as np
from glob import glob

from tensorflow import ConfigProto, Session
from tensorflow.python import keras
from tensorflow.keras.losses import crossentropy
from tensorflow.python.keras.layers import Input, Dense
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.merge import concatenate

from generators import get_gens

class KerasPilot(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car.
    '''
    def __init__(self):
        self.model = None
        self.optimizer = "adam"

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def train(self, train_gen, val_gen, train_steps, val_steps,
              saved_model_path, epochs=100, steps=100, 
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

    def get_features(self, img_arr):
        lp = 0 # learning_phase
        #img_arr = img_arr.reshape((1,) + img_arr.shape)
        inp = self.model.input
        out = self.model.layers[14].output
        func = K.function([inp, K.learning_phase()], out)
        layer_out = func([img_arr, lp])
        return layer_out



################################################################################

class KerasLinear(KerasPilot):
    '''
    The KerasLinear pilot uses one neuron to output a continous value via the 
    Keras Dense layer with linear activation. One each for steering and throttle.
    The output is not bounded.
    '''
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3), roi_crop=(0, 0), model=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        if model is not None:
            self.model = model
        else:
            assert False, "You need to provide a model, dipshit!"

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

################################################################################

def loss(y_true, y_pred):
    gt_angle, gt_throttle, gt_is_sim, point_5 = y_true
    pred_angle, pred_throttle, pred_is_sim, point_5 = y_pred

    loss  = K.mean(K.square(pred_angle-gt_angle))
    loss += K.mean(K.square(pred_throttle-gt_throttle))
    loss += K.mean(K.square(pred_is_sim-gt_is_sim))
    loss += K.mean(K.square(pred_smoosh - point_5))
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
    def __init__(self, input_shape=(120, 160, 3), model=None, *args, **kwargs):
        super(KerasLinearAdversarialDistributionSmoosher, self).__init__(*args, **kwargs)
        self.model = model

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=loss, metrics={"control_only_metric" : control_only_metric})

    def train(self, train_gen, val_gen, train_steps, val_steps,
              saved_model_path, epochs=100, steps=100,
              verbose=1, min_delta=.0005, patience=5):

        copy_weights_callback = CopyWeights(self.model)

        return super.train(train_gen, val_gen, train_steps, val_steps,
                           saved_model_path, epochs=epochs, steps=steps,
                           verbose=verbose, min_delta=min_delta,
                           patience=patience,
                           monitor="val_control_only_metric,
                           other_callbacks = [copy_weights_callback]):

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        sim_or_real = outputs[2]
        return steering[0][0], throttle[0][0]

################################################################################

def loss(y_true, y_pred):
    gt_angle, gt_throttle, gt_is_sim, point_5 = y_true
    pred_angle, pred_throttle, pred_is_sim, point_5 = y_pred

    loss  = K.mean(K.square(pred_angle-gt_angle))
    loss += K.mean(K.square(pred_throttle-gt_throttle))
    
    is_sim_loss = K.
    loss += K.mean(K.square(pred_is_sim-gt_is_sim))
    loss += K.mean(K.square(pred_smoosh - point_5))
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
    def __init__(self, input_shape=(120, 160, 3), model=None, *args, **kwargs):
        super(KerasLinearAdversarialDistributionSmoosher, self).__init__(*args, **kwargs)
        self.model = model

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=loss, metrics={"control_only_metric" : control_only_metric})

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        sim_or_real = outputs[2]
        return steering[0][0], throttle[0][0]


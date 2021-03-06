import os
import numpy as np
from glob import glob
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.merge import concatenate
from callbacks import CopyWeights

from generators import get_gens

__all__ = ["DefaultLinear", "CrossentropySmoosh", "LinearSmoosh"]

class KerasPilot(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car.
    '''
    def __init__(self, model, save_dir):
        self.model = model
        self.optimizer = "adam"
        self.save_dir = save_dir

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        assert False, "Not implemented :-("

    def compile(self):
        assert False, "Not implemented :-("

    def train(self, train_gen, val_gen, train_steps, val_steps,
              epochs=100, verbose=1, min_delta=.0005, patience=5,
              monitor="val_loss", use_early_stop=True, save_nth=None,
              other_callbacks = []):

        """
        train_gen: generator that yields an array of images an array of

        """
        callbacks_list = other_callbacks
        saved_model_path = os.path.join(self.save_dir,"latest.h5")
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                    monitor=monitor,
                                                    verbose=verbose,
                                                    save_best_only=True,
                                                    mode='min')
        callbacks_list.append(save_best)

        #checkpoint to save model every nth epoch
        if save_nth is not None:
            path = os.path.join(self.save_dir, 'weights{epoch:03d}.h5')
            save_nth_callback = keras.callbacks.ModelCheckpoint(path,
                                                        period=save_nth,
                                                        monitor=monitor,
                                                        verbose=verbose,
                                                        mode='min')
            callbacks_list.append(save_nth_callback)

        #stop training if the validation error stops improving.
        if use_early_stop:
            early_stop = keras.callbacks.EarlyStopping(monitor=monitor,
                                                   min_delta=min_delta,
                                                   patience=patience,
                                                   verbose=verbose,
                                                   mode='min')
            callbacks_list.append(early_stop)
        import pdb; pdb.set_trace()
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
        # Layer 14 is the final fully connected layer before splitting into 
        # different heads.
        lp = 0 # learning_phase
        inp = self.model.input
        out = self.model.layers[14].output
        func = K.function([inp, K.learning_phase()], out)
        features = func([img_arr, lp])
        return features[0].tolist()


################################################################################

from layers import default_n_linear
class DefaultLinear(KerasPilot):
    '''
    The DefaultLinear pilot uses one neuron to output a continous value via the 
    Keras Dense layer with linear activation. One each for steering and throttle.
    The output is not bounded.
    '''

    def __init__(self, save_dir):
        model = default_n_linear()
        super(DefaultLinear, self).__init__(model, save_dir)

    def compile(self):
        loss = "mse"
        self.model.compile(optimizer=self.optimizer,
                loss=loss)

################################################################################

from layers import smoosh_linear
from losses import is_sim_linear_loss
class LinearSmoosh(KerasPilot):
    '''
    The DefaultLinear pilot uses one neuron to output a continous value via the
    Keras Dense layer with linear activation. One each for steering and throttle.
    The output is not bounded.
    '''

    def __init__(self, save_dir):
        model = smoosh_linear()
        super(LinearSmoosh, self).__init__(model, save_dir)

    def compile(self):
        loss = is_sim_linear_loss()
        self.model.compile(optimizer=self.optimizer,
                           loss=loss)
        print(f"Availiable metrics: {self.model.metrics_names}")

    def train(self, train_gen, val_gen, train_steps, val_steps,
              epochs=100, verbose=1, min_delta=.0005, patience=5,
              use_early_stop=True, save_nth=None):

        copy_weights_callback = CopyWeights(self)
        return super().train(train_gen, val_gen, train_steps, val_steps,
                             epochs=epochs,
                             verbose=verbose,
                             min_delta=min_delta,
                             patience=patience,
                             monitor="val_steering_output_loss",
                             use_early_stop=use_early_stop,
                             save_nth=save_nth,
                             other_callbacks = [copy_weights_callback])


################################################################################

from layers import smoosh_classification
from losses import is_sim_categorical_loss
class CrossentropySmoosh(KerasPilot):
    '''

    '''

    def __init__(self, save_dir):
        model = smoosh_classification()
        super(CrossentropySmoosh, self).__init__(model, save_dir)

    def compile(self):
        loss = is_sim_categorical_loss()
        self.model.compile(optimizer=self.optimizer,
                           loss=loss)

    def train(self, train_gen, val_gen, train_steps, val_steps,
              epochs=100, verbose=1, min_delta=.0005, patience=5,
              use_early_stop=True, save_nth=None):

        copy_weights_callback = CopyWeights(self)
        return super().train(train_gen, val_gen, train_steps, val_steps,
                           epochs=epochs,
                           verbose=verbose,
                           min_delta=min_delta,
                           patience=patience,
                           monitor="val_steering_output_loss",
                           use_early_stop=use_early_stop,
                           save_nth=save_nth,
                           other_callbacks = [copy_weights_callback])

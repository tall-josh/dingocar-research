from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.python import keras
from tensorflow.keras.losses import binary_crossentropy as bce
from tensorflow.python.keras.layers import Input, Dense
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.merge import concatenate

def conv_layers():
    input_shape = (120, 160, 3)

    drop = 0.1

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)
    return img_in, x

################################################################################

def default_n_linear():
    img_in, x  = conv_layers()
    steering   = Dense(1, activation="linear", name="steering_output")(x)
    throttle   = Dense(1, activation="linear", name="throttle_output")(x)

    outputs = [steering, throttle]
    model = Model(inputs=[img_in], outputs=outputs)
    return model

################################################################################

def smoosh_linear():
    img_in, x   = conv_layers()
    steering    = Dense(1, activation="linear", name="steering_output")(x)
    throttle    = Dense(1, activation="linear", name="throttle_output")(x)
    smoosh      = Dense(1, activation="sigmoid", name="smoosh_output")(x)
    x_stop_grad = Lambda(lambda x: K.stop_gradient(x))(x)

    # real=0, sim=1
    is_sim      = Dense(1, activation="sigmoid", name="is_sim_output")(x_stop_grad)

    outputs = [steering, throttle, is_sim, smoosh]
    model = Model(inputs=[img_in], outputs=outputs)
    return model

################################################################################

def smoosh_classification():
    img_in, x   = conv_layers()
    steering    = Dense(1, activation="linear", name="steering_output")(x)
    throttle    = Dense(1, activation="linear", name="throttle_output")(x)
    smoosh      = Dense(2, activation="softmax", name="smoosh_output")(x)
    x_stop_grad = Lambda(lambda x: K.stop_gradient(x))(x)

    # real=0, sim=1
    is_sim      = Dense(2, activation="softmax", name="is_sim_output")(x_stop_grad)

    outputs = [steering, throttle, is_sim, smoosh]
    model = Model(inputs=[img_in], outputs=outputs)
    return model

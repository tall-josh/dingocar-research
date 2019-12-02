from tensorflow.python import keras

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

class ControlOnlyEarlyStop(keras.callbacks.Callback):

    def __init__(self):
        pass
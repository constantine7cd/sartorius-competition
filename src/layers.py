import tensorflow as tf


class Identity(tf.keras.layers.Layer):
    def call(self, x):
        return x

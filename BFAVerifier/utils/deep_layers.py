from tensorflow.keras.layers import Layer
# from backward_pkg.utils_quadapter import *
import tensorflow as tf
from utils.quantization_util import *

if tf.__version__.startswith("1.") or tf.__version__.startswith("0."):
    raise ValueError("Please upgrade to TensorFlow 2.x")


class DeepModel(tf.keras.Model):
    def __init__(
            self,
            layers,
            activation="relu",
            dropout_rate=0.0,
            last_layer_signed=False,
    ):

        super(DeepModel, self).__init__()
        self.dense_layers = []
        for i, l in enumerate(layers):
            if type(l) == int:
                signed = (
                    True if i == len(layers) - 1 else False
                )
                last_layer = (True if i == len(layers) - 1 else False)
                self.dense_layers.append(
                    DeepDense(
                        output_dim=l,
                        signed_output=signed,
                        activation=activation,
                        if_output_layer=last_layer,
                    )
                )
            else:
                raise ValueError("Unexpected type {} ({})".format(type(l), str(l)))
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.flatten_layer = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self._input_shape = input_shape
        super(DeepModel, self).build(
            input_shape
        )

    def call(self, inputs, **kwargs):
        x = inputs / 255
        x = tf.cast(x, tf.float32)
        # print(inputs)
        for c in self.dense_layers:
            if self.dropout_rate > 0.0:
                x = self.dropout_layer(x)
            # print(x)
            x = c(x)
        return x


class EranDeepModel(tf.keras.Model):
    def __init__(
            self,
            layers,
            dropout_rate=0.0,
            last_layer_signed=False,
    ):

        super(EranDeepModel, self).__init__()
        self.dense_layers = []
        for i, l in enumerate(layers):
            if type(l) == int:
                #  each layer is followed by a relu
                self.dense_layers.append(
                    DeepDense(
                        output_dim=l,
                        signed_output=False,
                        if_output_layer=False
                    )
                )
            else:
                raise ValueError("Unexpected type {} ({})".format(type(l), str(l)))
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.flatten_layer = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self._input_shape = input_shape
        super(EranDeepModel, self).build(
            input_shape
        )

    def call(self, inputs, **kwargs):
        x = inputs / 255  # float32 input
        x = tf.cast(x, tf.float32)
        for c in self.dense_layers:
            if self.dropout_rate > 0.0:
                x = self.dropout_layer(x)
            print(x)
            x = c(x)
        return x


class DeepLayer(Layer):
    def __init__(self, input_bits=None, quantization_config=None, **kwargs):
        super(DeepLayer, self).__init__(**kwargs)


class DeepDense(DeepLayer):
    def __init__(
            self,
            output_dim,
            activation="relu",
            signed_output=False,
            if_output_layer=False,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
            **kwargs
    ):
        self.units = output_dim
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.signed_output = signed_output
        self.if_output_layer = if_output_layer
        super(DeepDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[1]), self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer=tf.keras.initializers.Constant(0.25),
            trainable=True,
            # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
        )

        super(DeepDense, self).build(
            input_shape
        )

    def call(self, x, training=None):
        y = tf.matmul(x, self.kernel) + self.bias
        if self.activation == "relu":
            y = deep_op_activation_relu(y, self.if_output_layer)
        elif self.activation == "sigmoid":
            y = deep_op_activation_sigmoid(y, self.if_output_layer)
        elif self.activation == "tanh":
            y = deep_op_activation_tanh(y, self.if_output_layer)
        else:
            raise ValueError("Unexpected activation function {}".format(self.activation))
        return y

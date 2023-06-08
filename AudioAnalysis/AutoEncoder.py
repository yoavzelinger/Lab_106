from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation
from keras import backend as k
from keras.optimizers import Adam
from keras.losses import MeanSquaredError as Mse
import numpy as np
import pickle
import os


def load_model(load_folder):
    parameters_path = os.path.join(load_folder, "parameters.pkl")
    with open(parameters_path, "rb") as f:
        parameters = pickle.load(f)
    autoencoder = Autoencoder(*parameters)
    weights_path = os.path.join(load_folder, "weights.h5")
    autoencoder.total_model.load_weights(weights_path)
    return autoencoder


class Autoencoder:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self._input_shape = input_shape
        self._model_input = Input(shape=input_shape, name="encoder_input")
        self._shape_before_bottleneck = None
        self.latent_space_dim = latent_space_dim

        self.encoder_filters = conv_filters
        self.encoder_kernels = conv_kernels
        self.encoder_strides = conv_strides

        # Reversing order for decoder
        self.decoder_filters = conv_filters[::-1]
        self.decoder_kernels = conv_kernels[::-1]
        self.decoder_strides = conv_strides[::-1]

        # Creating the model
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

        # Adding it all together
        self.total_model = self._create_total_model()

    def _create_total_model(self):
        model_input = self._model_input
        encoder_output = self.encoder(model_input)
        model_output = self.decoder(encoder_output)
        return Model(model_input, model_output, name="autoencoder")

    def _apply_convolutions(self, convolution_input, convolutions_details, direction_str):
        current_input = convolution_input
        for layer_index, layer_info in enumerate(convolutions_details):
            current_input = self._apply_convolution_layer(current_input, layer_index, layer_info, direction_str)
        return current_input

    def _apply_convolution_layer(self, layer_input, layer_index, layer_info, direction_str):
        conv_filter, kernel, stride = layer_info
        layer_desc = direction_str + "_{0}_layer_" + str(layer_index + 1)
        conv_func = Conv2D if direction_str == "encoder" else Conv2DTranspose
        current_conv = conv_func(filters=conv_filter, kernel_size=kernel, strides=stride,
                                 padding="same", name=layer_desc.format("conv"))
        current_relu = ReLU(name=layer_desc.format("relu"))
        current_bn = BatchNormalization(name=layer_desc.format("bn"))
        current_input = current_conv(layer_input)
        current_input = current_relu(current_input)
        return current_bn(current_input)

    # ENCODER PART
    def _create_encoder(self):
        encoder_input = self._model_input
        convoluted_input = self._encoder_convolution(encoder_input)
        bottleneck = self._apply_bottleneck(convoluted_input)
        return Model(encoder_input, bottleneck, name="encoder")

    def _encoder_convolution(self, encoder_input):
        convolution_details = zip(self.encoder_filters, self.encoder_kernels, self.encoder_strides)
        return self._apply_convolutions(encoder_input, convolution_details, "encoder")

    def _apply_bottleneck(self, bottleneck_input):
        self._shape_before_bottleneck = k.int_shape(bottleneck_input)[1:]
        flatten_input = Flatten()(bottleneck_input)
        return Dense(self.latent_space_dim, name="encoder_output")(flatten_input)

    # DECODER PART
    def _create_decoder(self):
        decoder_input = Input(shape=self.latent_space_dim, name="decoder_input")
        dense_reshape = self._apply_dense_reshape(decoder_input)
        reverted_convolution = self._decoder_convolutions(dense_reshape)
        decoder_output = Activation("sigmoid", name="sigmoid_layer")(reverted_convolution)
        return Model(decoder_input, decoder_output, name="decoder")

    def _apply_dense_reshape(self, dense_input):
        neurons_count = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(neurons_count, name="decoder_dense")(dense_input)
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _decoder_convolutions(self, reverting_input):
        convolution_details = zip(self.decoder_filters, self.decoder_kernels, self.decoder_strides)
        return self._apply_convolutions(reverting_input, convolution_details, "decoder")

    def compile_model(self, learning_rate=0.0001):
        optimizer, loss = Adam(learning_rate=learning_rate), Mse()
        self.total_model.compile(optimizer=optimizer, loss=loss)

    def train_model(self, x_train, batch_size, epochs_count):
        self.total_model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=epochs_count)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.total_model.summary()

    def save_model(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        parameters = [self._input_shape, self.encoder_filters, self.encoder_kernels, self.encoder_strides,self.latent_space_dim]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
        save_path = os.path.join(save_folder, "weights.h5")
        self.total_model.save_weights(save_path)

# if __name__ == "__main__":
#     autoencoder = Autoencoder(
#         input_shape=(28, 28, 1),
#         conv_filters=(32, 64, 64, 64),
#         conv_kernels=(3, 3, 3, 3),
#         conv_strides=(1, 2, 2, 1),
#         latent_space_dim=2
#     )
#     autoencoder.summary()

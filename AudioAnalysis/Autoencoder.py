import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import models, optimizers, callbacks
from keras.activations import relu, sigmoid, softmax
from keras.layers import Convolution1D, Dense, Input, Conv1DTranspose, Dropout, MaxPool1D, GlobalMaxPool1D, Reshape, \
    UpSampling1D
from keras.regularizers import l1_l2
import os


class Autoencoder:
    def __init__(self, sample_length_ms=5000, frame_rate=48000, bottleneck_size=0.4, l1_value=0.01, l2_value=0.02):
        self.input_size = int(sample_length_ms * frame_rate / 1000)
        self.model_input = Input(shape=(self.input_size, 1))

        self._dense_layers = self._get_dense_layers(bottleneck_size)
        self._regularization = l1_l2(l1=l1_value, l2=l2_value)

        self.shape_before_GlobalMaxPool = None

        self.encoder = self._create_encoder()
        # self.encoder = self._new_encoder()
        encoder_output_shape = self.encoder.output_shape[1:]

        self.decoder = self._create_decoder(encoder_output_shape)
        # self.decoder = self._new_decoder()

        self.auto_encoder = self._create_total_model()

        tf.random.set_seed(0)

    def _get_dense_layers(self, bottleneck_size):
        dense_layers = [self.input_size]
        while dense_layers[-1] / self.input_size > bottleneck_size:
            dense_layers.append(dense_layers[-1] // 2 + 1)
        print(dense_layers)
        return dense_layers

    def _create_total_model(self):
        model_input = self.model_input
        encoder_output = self.encoder(model_input)
        model_output = self.decoder(encoder_output)
        return models.Model(model_input, model_output, name="autoencoder")

    def _new_encoder(self):
        x = self.model_input
        for dense_layer in self._dense_layers[1: -1]:
            x = Dense(units=dense_layer, activation=relu, kernel_regularizer=self._regularization)(x)
        out = Dense(units=self._dense_layers[-1], activation=sigmoid, kernel_regularizer=self._regularization)(x)
        return models.Model(inputs=self.model_input, outputs=out, name="encoder")

    def _new_decoder(self):
        latent_inputs = Input(self._dense_layers[-1])
        x = latent_inputs
        for dense_layer in reversed(self._dense_layers[1: -1]):
            x = Dense(units=dense_layer, activation=relu, kernel_regularizer=self._regularization)(x)
        out = Dense(units=self.input_size, activation=sigmoid, kernel_regularizer=self._regularization)(x)
        return models.Model(inputs=latent_inputs, outputs=out, name="decoder")

    #   Encoder
    def _create_encoder(self):
        x = self.model_input
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        # x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        # x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        # x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        self.shape_before_GlobalMaxPool = x.shape

        x = GlobalMaxPool1D()(x)
        # x = Dropout(rate=0.2)(x)

        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)

        out = x

        return models.Model(inputs=self.model_input, outputs=out, name="encoder")

    #   Decoder
    def _create_decoder(self, encoder_output_shape):
        latent_inputs = Input(shape=encoder_output_shape)

        x = latent_inputs

        x = Dense(64, activation='relu')(latent_inputs)
        x = Dense(1028, activation='relu')(x)   # TODO - Check how to modify to make it run faster
        x = Dense(np.prod(self.shape_before_GlobalMaxPool[1:]), activation='relu')(x)
        x = Reshape(self.shape_before_GlobalMaxPool[1:])(x)

        # x = Dropout(rate=0.2)(x)
        x = Conv1DTranspose(256, 3, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(256, 3, activation='relu', padding='valid')(x)

        x = UpSampling1D(4)(x)
        # x = Dropout(rate=0.1)(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)

        x = UpSampling1D(4)(x)
        # x = Dropout(rate=0.1)(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)

        x = UpSampling1D(16)(x)
        # x = Dropout(rate=0.1)(x)
        x = Conv1DTranspose(16, 9, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(16, 9, activation='relu', padding='valid')(x)

        out = Conv1DTranspose(1, 49, activation='sigmoid', padding='valid')(x)

        return models.Model(inputs=latent_inputs, outputs=out, name="decoder")

    def compile_model(self, learning_rate=0.01):
        self.auto_encoder.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', run_eagerly=True)
        print(self.auto_encoder.summary())

    def train_model(self, x_train, batch_size=32, epochs_count=100, validation_size=0.8):
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
        callback_early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        # tensorboard_callback = callbacks.TensorBoard(log_dir="model/" + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)  # TODO - Check if necessary
        # self.auto_encoder.fit(x_train_split, x_train_split, batch_size=batch_size, epochs=epochs_count, validation_data=(x_val_split, x_val_split), callbacks=[reduce_lr]) # Old train
        start_time = datetime.now()
        history_time = self.auto_encoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs_count, validation_split=0.2, callbacks=[reduce_lr, callback_early_stop])
        end_time = datetime.now()
        print(f"Train time: {end_time - start_time}")
        with open('history_time.pkl', 'wb') as f:
            pickle.dump(history_time, f)
        # plt.plot(history_time.history['loss'], label='Training Loss')
        # plt.plot(history_time.history['val_loss'], label='Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

        # plt.plot(history_time.history['accuracy'], label='Training Accuracy')
        # plt.plot(history_time.history['val_accuracy'], label='Validation Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()

    def predict(self, x):
        return self.auto_encoder.predict(x)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.auto_encoder.summary()

    def save_model(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, "encoder_weights.h5")
        self.encoder.save_weights(save_path)
        save_path = os.path.join(save_folder, "decoder_weights.h5")
        self.decoder.save_weights(save_path)
        save_path = os.path.join(save_folder, "auto_encoder_weights.h5")
        self.auto_encoder.save_weights(save_path)


if __name__ == "__main__":
    t = Autoencoder()
    t.summary()

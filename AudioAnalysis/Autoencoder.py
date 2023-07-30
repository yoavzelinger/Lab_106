import numpy as np
from keras import models, optimizers, callbacks
from keras.activations import relu, softmax
from keras.layers import (Convolution1D, Dense, GlobalMaxPool1D, Input, MaxPool1D)
from keras.layers import (Conv1DTranspose, Reshape, UpSampling1D, Dropout)
import os


class Autoencoder:
    def __init__(self, sample_length_ms=5000, frame_rate=48000, n_classes=2):
        # TODO - Check if n_classes is two in our case
        self.input_length = int(sample_length_ms * frame_rate / 1000)
        self.model_input = Input(shape=(self.input_length, 1))

        self.n_classes = n_classes

        self.shape_before_GlobalMaxPool = None

        self.encoder = self._create_encoder()

        encoder_output_shape = self.encoder.output_shape[1:]

        self.decoder = self._create_decoder(encoder_output_shape)

        self.auto_encoder = self._create_total_model()

    def _create_total_model(self):
        model_input = self.model_input
        encoder_output = self.encoder(model_input)
        model_output = self.decoder(encoder_output)
        return models.Model(model_input, model_output, name="autoencoder")

    #   Encoder
    def _create_encoder(self):
        x = self.model_input
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        self.shape_before_GlobalMaxPool = x.shape

        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)

        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)  # TODO - Check how to modify to make it run faster
        out = Dense(self.n_classes, activation=softmax)(x)

        return models.Model(inputs=self.model_input, outputs=out, name="encoder")

    #   Decoder
    def _create_decoder(self, encoder_output_shape):
        latent_inputs = Input(shape=encoder_output_shape)

        x = latent_inputs

        x = Dense(64, activation='relu')(latent_inputs)
        x = Dense(1028, activation='relu')(x)   # TODO - Check how to modify to make it run faster
        x = Dense(np.prod(self.shape_before_GlobalMaxPool[1:]), activation='relu')(x)
        x = Reshape(self.shape_before_GlobalMaxPool[1:])(x)

        x = Dropout(rate=0.2)(x)
        x = Conv1DTranspose(256, 3, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(256, 3, activation='relu', padding='valid')(x)

        x = UpSampling1D(4)(x)
        x = Dropout(rate=0.1)(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)

        x = UpSampling1D(4)(x)
        x = Dropout(rate=0.1)(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(32, 3, activation='relu', padding='valid')(x)

        x = UpSampling1D(16)(x)
        x = Dropout(rate=0.1)(x)
        x = Conv1DTranspose(16, 9, activation='relu', padding='valid')(x)
        x = Conv1DTranspose(16, 9, activation='relu', padding='valid')(x)

        out = Conv1DTranspose(1, 49, activation='sigmoid', padding='valid')(x)

        return models.Model(inputs=latent_inputs, outputs=out, name="decoder")

    def create_dummy_encoder(self):
        x = GlobalMaxPool1D()(self.model_input)
        out = Dense(self.n_classes, activation=softmax)(x)

        return models.Model(inputs=self.model_input, outputs=out)

    def create_dummy_decoder(self, encoder_output_shape):
        latent_inputs = Input(shape=encoder_output_shape)

        x = latent_inputs

        x = Dense(np.prod(self.shape_before_GlobalMaxPool[1:]), activation='relu')(x)
        out = Reshape(self.shape_before_GlobalMaxPool[1:])(x)

        return models.Model(inputs=self.model_input, outputs=out)

    def compile_model(self, learning_rate=0.001):
        self.auto_encoder.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', run_eagerly=True)

    def train_model(self, x_train, batch_size=32, epochs_count=1, validation_size=0.8):
        split_idx = int(validation_size * len(x_train))
        x_train_split = x_train[:split_idx]
        x_val_split = x_train[split_idx:]
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
        self.auto_encoder.fit(x_train_split, x_train_split, batch_size=batch_size, epochs=epochs_count, validation_data=(x_val_split, x_val_split), callbacks=[reduce_lr])

    def predict(self, x):
        return self.auto_encoder.predict(x)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.auto_encoder.summary()

    def save_model(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, "weights.h5")
        self.auto_encoder.save_weights(save_path)


if __name__ == "__main__":
    t = Autoencoder()
    t.summary()

import NoiseRemoval
from AutoEncoder import Autoencoder
import os
import numpy as np

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150
INPUT_FOLDER = ""
WINDOW = 5000    # ms
OVERLAP = 1000  # ms


def receive_clean_audio(input_path):
    noise_remover = NoiseRemoval.NoiseRemover()
    return noise_remover.apply_filter(input_path)[0] # Clean cell


def window_time_array(audio, window_ms, overlap_ms):
    audio_array, frame_rate = audio.get_array_of_samples(), audio.frame_rate
    window_size = frame_rate * (window_ms / 1000)
    overlap_jump = window_size - frame_rate * (overlap_ms / 1000)
    freq_windows = []
    for start_frame in range(0, len(audio_array), overlap_jump):
        end_frame = min(start_frame + window_size, len(audio_array))
        time_window = audio_array[start_frame, end_frame]
        freq_windows.append(np.fft.fft(time_window))
    return np.array(freq_windows)


def create_train(input_folder, window_ms, overlap_ms):
    train = []
    for root, _, file_names in os.walk(input_folder):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            clean_audio = receive_clean_audio(file_path)
            frame_seperated = window_time_array(clean_audio, window_ms, overlap_ms)
            train.append(frame_seperated)
    return np.array(train)


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.compile_model(LEARNING_RATE)
    x_train = create_train(INPUT_FOLDER, WINDOW, OVERLAP)
    autoencoder.train_model(x_train, BATCH_SIZE, EPOCHS)
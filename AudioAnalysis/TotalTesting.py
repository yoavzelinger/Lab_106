import sys

import NoiseRemoval
from Autoencoder import Autoencoder
import os
import numpy as np
import pickle

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

LEARNING_RATE = 0.0005
BATCH_SIZE = 3
EPOCHS = 1
# INPUT_FOLDER = "C:\\Users\\yoavz\\Documents\\Inbal_Project\\Testing"
WINDOW = 5000  # ms
OVERLAP = 1000  # ms


def receive_clean_audio(input_path):
    noise_remover = NoiseRemoval.NoiseRemover()
    return noise_remover.apply_filter(input_path)[0]  # Clean cell


def window_time_array(audio, window_ms, overlap_ms):
    audio_array, frame_rate = audio.get_array_of_samples(), audio.frame_rate
    window_size = int(frame_rate * (window_ms / 1000))
    overlap_jump = int(window_size - frame_rate * (overlap_ms / 1000))
    freq_windows = []
    for start_frame in range(0, len(audio_array), overlap_jump):
        end_frame = min(start_frame + window_size, len(audio_array))
        time_window = audio_array[start_frame: end_frame]
        freq_window = np.fft.fft(time_window).real.tolist()
        if len(freq_window) == window_size:
            freq_windows.append(np.transpose(np.array([[freq_window]]), axes=(0, 2, 1)))
    return freq_windows


def create_train(input_folder, window_ms, overlap_ms):
    train = []
    for root, _, file_names in os.walk(input_folder):
        for file_name in file_names:
            if file_name[-3:].upper() != "AAC":
                continue
            file_path = os.path.join(root, file_name)
            clean_audio = receive_clean_audio(file_path)
            frame_seperated = window_time_array(clean_audio, window_ms, overlap_ms)
            train += frame_seperated
    return np.array(train, dtype=np.float64)


if __name__ == "__main__":
    INPUT_FOLDER = sys.argv[1]
    x_train = create_train(INPUT_FOLDER, WINDOW, OVERLAP)
    current = x_train[0]

    print("Finished creating x_train")
    autoencoder = Autoencoder()
    print("Finished creating model")
    autoencoder.compile_model()
    print("Finished compiling model")
    autoencoder.train_model(x_train[0], BATCH_SIZE, EPOCHS)
    print("Finished trying model")
    origin_sample = x_train[0][0][0]
    predicted_sample = autoencoder.predict(x_train[0][0][0])
    print("Saving files")
    with open('origin.pkl', 'wb') as f:
        pickle.dump(origin_sample, f)
    with open('predicted.pkl', 'wb') as f:
        pickle.dump(origin_sample, f)
    print("Done")
    # for a, b in zip(origin_sample, predicted_sample):
    #     print(f"Origin: {a}, Prediction: {b}")

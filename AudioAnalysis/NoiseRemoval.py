import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt


def input_to_audio_array(file_path, file_format="wav", start_time=0, end_time=-1):
    audio = AudioSegment.from_file(file_path, format=file_format)
    audio.get_array_of_samples()
    start_time = min(max(0, start_time), len(audio)) * 1000
    end_time = min(max(start_time, end_time), len(audio)) * 1000
    if start_time == end_time:
        end_time = len(audio)
    return audio[start_time: end_time]


def samples_to_output(samples_array, output_path):
    audio_file = AudioSegment(samples_array.tobytes(), frame_rate=48000, sample_width=2, channels=1)
    audio_file.export(output_path, format="wav")


def get_frequencies_magnitudes(audio, scaling=1):
    samples = audio.get_array_of_samples()
    frequencies = np.fft.fftfreq(len(samples), 1 / audio.frame_rate)
    magnitudes = np.fft.fft(samples) * scaling
    return frequencies, magnitudes


def get_samples(magnitudes):
    return np.fft.ifft(magnitudes).real.astype(np.int16)


def apply_mask(frequencies, sample_rate, ranges_to_remove):
    frequency_resolution = sample_rate / len(frequencies)
    mask = np.zeros_like(frequencies)
    for lower_freq, upper_freq in ranges_to_remove:
        lower_index = int(lower_freq / frequency_resolution)
        upper_index = int(upper_freq / frequency_resolution)
        mask[lower_index: upper_index] = 1
    return mask


def filer_frequencies(input_path, output_path, frequencies_to_keep):
    audio = input_to_audio_array(input_path)
    frequencies, magnitudes = get_frequencies_magnitudes(audio)
    remove_mask = apply_mask(frequencies, audio.frame_rate, frequencies_to_keep)
    filtered_magnitudes = magnitudes * remove_mask
    filtered_samples = get_samples(filtered_magnitudes)
    samples_to_output(filtered_samples, output_path)


def plot_frequency_domain(frequencies, magnitudes):
    plt.plot(np.abs(frequencies), np.abs(magnitudes))
    plt.show()


def get_complement_frequencies(frequencies_ranges, gap=24000):
    complement_frequencies = []
    frequencies_ranges.sort()
    current_freq = 0
    for freq_start, freq_end in frequencies_ranges:
        if current_freq != freq_start:
            complement_frequencies.append((current_freq, freq_start))
        current_freq = freq_end
    if current_freq < gap:
        complement_frequencies.append((current_freq, gap))
    return complement_frequencies


if __name__ == "__main__":
    input_file = "C:\\Users\\yoavz\\Desktop\\trimmed.wav"
    wind_output = "C:\\Users\\yoavz\\Desktop\\wind.wav"
    clear_output = "C:\\Users\\yoavz\\Desktop\\clear.wav"
    wind_frequencies = [(65, 500), (8000, 9000), (9100, 9300), (15780, 24000)]
    filer_frequencies(input_file, wind_output, wind_frequencies)
    keeping_frequencies = get_complement_frequencies(wind_frequencies)
    filer_frequencies(input_file, clear_output, keeping_frequencies)

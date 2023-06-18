import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt


class NoiseRemover:
    def __init__(self):
        self.wind_frequencies = [(65, 500), (8000, 9000), (9100, 9300), (15780, 24000)]
        self.wind_frequencies.sort()
        self.clean_frequencies = self._get_complement_frequencies()

    def _get_complement_frequencies(self, gap=24000):
        complement_frequencies = []
        current_freq = 0
        for freq_start, freq_end in self.wind_frequencies:
            if current_freq != freq_start:
                complement_frequencies.append((current_freq, freq_start))
            current_freq = freq_end
        if current_freq < gap:
            complement_frequencies.append((current_freq, gap))
        return complement_frequencies

    def apply_filter(self, input_path):
        audio = self._input_to_audio_array(input_path)
        clean_audio = self._filer_frequencies(audio, self.clean_frequencies)
        wind_audio = self._filer_frequencies(audio, self.wind_frequencies)
        return clean_audio, wind_audio

    def _input_to_audio_array(self, file_path, file_format="aac", start_time=0, end_time=-1):
        audio = AudioSegment.from_file(file_path, format=file_format)
        audio.get_array_of_samples()
        start_time = min(max(0, start_time), len(audio)) * 1000
        end_time = min(max(start_time, end_time), len(audio)) * 1000
        if start_time == end_time:
            end_time = len(audio)
        return audio[start_time: end_time]

    def _filer_frequencies(self, audio, frequencies_to_keep):
        frequencies, magnitudes = self._get_frequencies_magnitudes(audio)
        remove_mask = self._apply_mask(frequencies, audio.frame_rate, frequencies_to_keep)
        filtered_magnitudes = magnitudes * remove_mask
        return self._get_audio(filtered_magnitudes)

    def _get_frequencies_magnitudes(self, audio, scaling=1):
        samples = audio.get_array_of_samples()
        frequencies = np.fft.fftfreq(len(samples), 1 / audio.frame_rate)
        magnitudes = np.fft.fft(samples) * scaling
        return frequencies, magnitudes

    def _get_audio(self, magnitudes):
        samples_array = np.fft.ifft(magnitudes).real.astype(np.int16)
        return AudioSegment(samples_array.tobytes(), frame_rate=48000, sample_width=2, channels=1)

    def _apply_mask(self, frequencies, sample_rate, ranges_to_remove):
        frequency_resolution = sample_rate / len(frequencies)
        mask = np.zeros_like(frequencies)
        for lower_freq, upper_freq in ranges_to_remove:
            lower_index = int(lower_freq / frequency_resolution)
            upper_index = int(upper_freq / frequency_resolution)
            mask[lower_index: upper_index] = 1
        return mask

    def _plot_frequency_domain(self, frequencies, magnitudes):
        plt.plot(np.abs(frequencies), np.abs(magnitudes))
        plt.show()

    def save_to_file(self, audio, output_path, output_format="aac"):
        audio.export(output_path, format=output_format)


if __name__ == "__main__":
    input_file = "C:\\Users\\yoavz\\Desktop\\trimmed.wav"
    noise_remover = NoiseRemover()
    clean_audio, wind_audio = noise_remover.apply_filter(input_file)
    wind_output = "C:\\Users\\yoavz\\Desktop\\wind.wav"
    clean_output = "C:\\Users\\yoavz\\Desktop\\clear.wav"
    noise_remover.save_to_file(clean_audio, clean_output)
    noise_remover.save_to_file(wind_audio, wind_output)

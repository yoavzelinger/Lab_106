from pydub import AudioSegment
import math
import struct

SEGMENT_SIZE = 100  # ms
WIDTH = 2   # 2*8 bit
CHANNELS = 1
BIT_RATE = 48000


def divide_file(file_path):
    sound = AudioSegment.from_file(file_path, format="raw", sample_width=WIDTH, channels=CHANNELS, frame_rate=BIT_RATE)
    segment_count = math.ceil(len(sound) / SEGMENT_SIZE)
    for segment_num in range(segment_count):
        segment_start = segment_num * SEGMENT_SIZE
        segment_end = min(segment_start + SEGMENT_SIZE, len(sound))
        yield sound[segment_start:segment_end]

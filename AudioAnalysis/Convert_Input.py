import ffmpeg
import sys
import os
from pydub import AudioSegment
import math
import struct


def convert_to_pcm(root_path, output_path):
    """
    Convert the input File to PCM
    :param root_path: path to source file
    :param output_path: path to the destination file
    """
    if output_path == "":
        output_path = root_path[:-3] + "aac"
    else:
        output_path = output_path + root_path[root_path.rfind('\\'):-3] + "aac"
    stream = ffmpeg.input(root_path)
    stream = ffmpeg.output(stream, output_path, acodec='aac', ac=1, ar=48000, audio_bitrate=45000)
    ffmpeg.run(stream)
    return output_path


def divide_file(file_path):
    SEGMENT_SIZE = 300000  # ms
    WIDTH = 2  # 2 * 8bit
    CHANNELS = 1
    BIT_RATE = 48000
    sound = AudioSegment.from_file(file_path, format="raw", sample_width=WIDTH, channels=CHANNELS, frame_rate=BIT_RATE)
    segment_count = math.ceil(len(sound) / SEGMENT_SIZE)
    for segment_num in range(segment_count):
        segment_start = segment_num * SEGMENT_SIZE
        segment_end = min(segment_start + SEGMENT_SIZE, len(sound))
        yield sound[segment_start:segment_end]


def convert_files(source_type: str, search_flag: chr, source_path: str, destination_path=""):
    if source_type.upper() not in ["MP4", "AAC"]:
        print(f"Invalid input type: {source_type}")
        return
    for input_path in files_gen(source_type, search_flag, source_path):
        convert_to_pcm(input_path, destination_path)
        # for i, segment in enumerate(divide_file(output_path)):
        #     segment.export(output_path[:-4] + f"_segment{i}.pcm", format="raw")


def files_gen(input_type: str, search_flag: chr, source_path: str) -> str:
    def from_type(file: str): return len(file) > 4 and file[-3:].upper() == input_type.upper()
    match search_flag:
        case 'f':
            if from_type(source_path):
                yield source_path
        case 'd':
            for file_path in os.scandir(source_path):
                if file_path.is_file() and from_type(file_path.path):
                    yield file_path.path
        case 'r':
            for path, subdir, files in os.walk(source_path):
                for file_name in files:
                    file_path = os.path.join(path, file_name)
                    if os.path.isfile(file_path) and from_type(file_path):
                        yield file_path
        case _:
            print(f"Invalid path type: {search_flag}")


if __name__ == "__main__":
    print("Welcome")
    if len(sys.argv) - 1 in range(3, 5):
        convert_files(*sys.argv[1:])

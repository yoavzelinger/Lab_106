import ffmpeg
import sys
import os
from CPM_Segments import divide_file


def convert_to_pcm(root_path, output_path):
    """
    Convert the input File to PCM
    :param root_path: path to source file
    :param output_path: path to the destination file
    """
    if output_path == "":
        output_path = root_path[:-3] + "pcm"
    else:
        output_path = output_path + root_path[root_path.rfind('\\'):-3] + "pcm"
    stream = ffmpeg.input(root_path)
    stream = ffmpeg.output(stream, output_path, format='s16le', acodec='pcm_s16le', ac=1, ar=48000, audio_bitrate=45000)
    ffmpeg.run(stream)
    return output_path


def convert_files(source_type: str, search_flag : chr, source_path : str, destination_path=""):
    if source_type.upper() not in ["MP4", "AAC"]:
        print(f"Invalid input type: {source_type}")
        return
    for input_path in files_gen(source_type, search_flag, source_path):
        output_path = convert_to_pcm(input_path, destination_path)
        for i, segment in enumerate(divide_file(output_path)):
            segment.export(output_path[:-4] + f"_segment{i}.pcm", format="raw")


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

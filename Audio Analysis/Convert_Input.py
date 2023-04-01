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


def convert_input(input_lst):
    """
    :param input_lst: [source type, flag {f=folder, d=directory, r=recursive}, source, destination (optional, def=same)]
    """
    input_type = input_lst[1].upper()
    print(input_type)
    if input_type not in ["MP4", "AAC"]:
        print(f"Invalid input type: {input_type}")
        return
    source_path = input_lst[3]
    dest_path = ""
    if len(input_lst) > 4:
        dest_path = input_lst[4]
    match sys.argv[2]:
        case 'f':   # File
            if source_path[-3:] != input_type:
                print("Source file extension doesn't match input type")
                return
            output_path = convert_to_pcm(source_path, dest_path)
            for i, segment in enumerate(divide_file(output_path)):
                segment.export(output_path[:-4] + f"_segment{i}.cpm", format="raw")
        case 'd':   # Directory
            for file_path in os.scandir(source_path):
                if file_path.is_file() and file_path.path[-3:].upper() == input_type:
                    output_path = convert_to_pcm(file_path.path, dest_path)
                    for i, segment in enumerate(divide_file(output_path)):
                        segment.export(output_path[:-4] + f"_segment{i}.cpm", format="raw")
        case 'r':   # Recursive
            for path, subdir, files in os.walk(source_path):
                for name in files:
                    file_path = os.path.join(path, name)
                    if os.path.isfile(file_path) and file_path[-3:].upper() == input_type:
                        output_path = convert_to_pcm(file_path, dest_path)
                        for i, segment in enumerate(divide_file(output_path)):
                            segment.export(output_path[:-4] + f"_segment{i}.cpm", format="raw")
        case _:
            print(f"Invalid path type: {input_lst[2]}")


if __name__ == "__main__":
    print("Welcome")
    convert_input(sys.argv)

import os
import shutil
from os.path import basename, splitext

source_folder = r"/Users/shashankgupta/Documents/Raredis/Dataset/Raw_data/Test/"
destination_folder = r"/Users/shashankgupta/Desktop/Pipeline_testing/input_text/"

# fetch all files
for file_name in os.listdir(source_folder):
    doc_name = splitext(basename(file_name))[1]
    if doc_name == ".txt":
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        #
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
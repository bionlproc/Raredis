# import os
# import tarfile
#
# source_dir = "/Users/shashankgupta/Downloads/epoch50"
# output_file = "/Users/shashankgupta/Downloads/epoch50/weights.tar.gz"
#
# # create a tarfile object with gzip compression
# with tarfile.open(output_file, "w:gz") as tar:
#     # Add all files and directories inside the source_dir to the tarfile
#     for item in os.listdir(source_dir):
#         tar.add(os.path.join(source_dir, item), arcname=os.path.basename(item))

import os
import tarfile

source_dir = "/Users/shashankgupta/Documents/Raredis/SODNER/trained_models/pubmedbert_uncased/model"
output_file = "/Users/shashankgupta/Documents/Raredis/SODNER/trained_models/pubmedbert_uncased/model.tar.gz"

# create a tarfile object with gzip compression
with tarfile.open(output_file, "w:gz") as tar:
    # Add all files and directories inside the source_dir to the tarfile
    for item in os.listdir(source_dir):
        tar.add(os.path.join(source_dir, item), arcname=os.path.basename(item))
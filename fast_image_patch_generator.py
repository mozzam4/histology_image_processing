import fast
import numpy as np
from skimage import io

#  Set the input file path and the output Folder path
input_file_path = r'C:\Users\mozza\Documents\test\M166A_issues2.ndpi'
output_folder_path = r'C:\Users\mozza\Documents\test\patches\M166A_issues2'

importer = fast.WholeSlideImageImporter.create(input_file_path)

tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

patchGenerator = fast.PatchGenerator.create(1024, 1024, level=0)\
    .connect(0, importer)\
    .connect(1, tissueSegmentation)

count = 0
for patch in fast.DataStream(patchGenerator):
    path_to_save_patch = output_folder_path + '\ ' + str(count) + '.jpg'
    patch = np.asanyarray(patch)
    io.imsave(path_to_save_patch, patch)
    count = count + 1
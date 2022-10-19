import fast
import numpy as np
import os
from scipy import ndimage as ndi
from skimage import io, feature, color
import random

dir = r'C:\Users\mozza\Documents\test\Few_patches\selected_images'
output_dir = r'C:\Users\mozza\Documents\test\Few_patches\selected_patches_small'
for filename in os.listdir(dir):



    input_file_path = os.path.join(dir, filename)
    output_file_dir = os.path.join(output_dir, filename)
    if not os.path.exists(output_file_dir):
        os.mkdir(output_file_dir)
        importer = fast.WholeSlideImageImporter.create(input_file_path)
        tissueSegmentation = fast.TissueSegmentation.create().connect(importer)
        patchGenerator = fast.PatchGenerator.create(512, 512, level=0)\
            .connect(0, importer)\
            .connect(1, tissueSegmentation)

        count = 0
        patch_list = []
        processed_patches = []
        for patch in fast.DataStream(patchGenerator):

            patch_list.append(patch)
        while count < 500:
            random_patch = random.randint(0, len(patch_list) - 1)
            if random_patch in processed_patches:
                continue
            # check if the patch fulfils the reqt
            img = np.asanyarray(patch_list[random_patch])
            img_gray = color.rgb2gray(img)
            img_blur = ndi.gaussian_filter(img_gray, 3)
            edges1 = feature.canny(img_blur, sigma=1)
            unique, counts = np.unique(edges1, return_counts=True)
            edge_to_image_ratio = counts[1] / counts[0]
            if edge_to_image_ratio < 0.01:
                continue

            processed_patches.append(random_patch)
            count = count + 1
            path_to_save_patch = output_file_dir + '\ ' + str(count) + '.jpg'
            io.imsave(path_to_save_patch, img)
            del img, img_gray, img_blur, edges1

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, feature, color
import itk
import os
# Normlaise all the patches based on reference image. The code is incomplete.


reference_image_filename = r'/home/mozzam/Downloads/Ref.png'
reference_image = itk.imread(reference_image_filename)
#reference_reader = itk.ImageFileReader.New(FileName=reference_image_filename)


dir = r'/home/mozzam/Documents/Few_patches/patches/R021A_reference_clean'
for filename in os.listdir(dir):
    print(filename)
    input_filename = os.path.join(dir, filename)
    # try:
    #     input_reader = itk.ImageFileReader.New(FileName=input_filename)
    #     spcn_filter = itk.StructurePreservingColorNormalizationFilter.New(Input=input_reader.GetOutput())
    #     spcn_filter.SetColorIndexSuppressedByHematoxylin(0)
    #     spcn_filter.SetColorIndexSuppressedByEosin(1)
    #     spcn_filter.SetInput(0, input_reader.GetOutput())
    #     spcn_filter.SetInput(1, reference_reader.GetOutput())
    #
    #     output_writer = itk.ImageFileWriter.New(spcn_filter.GetOutput())
    #     output_writer.SetInput(spcn_filter.GetOutput())
    #     dir_to_save = r'C:\Users\mozza\Documents\test\patches\M143A1_issues_normalised'
    #     path_to_save = os.path.join(dir_to_save, filename)
    #     output_writer.SetFileName(path_to_save)
    #     output_writer.Write()
    #
    # except Exception as X:
    #     print(X)

    with open(input_filename, 'r') as f:
        # use canny ede detector to check if the file has any issues as per ECHLE2022 preprocessing

        #Getting the canny edges
        #threshold = 4
        img = io.imread(input_filename)
        img_gray = color.rgb2gray(img)
            #img_gray[img_gray < threshold] = 0
            #img_gray[img_gray >= threshold] = 255
        img_blur = ndi.gaussian_filter(img_gray, 3)
        edges1 = feature.canny(img_blur, sigma=1)
        #edges2 = feature.canny(img_blur, sigma=4)

        #io.imsave(r'/home/mozzam/Documents/Few_patches/patches/R021A_reference_clean/edges1.jpg', edges1)
        #io.imsave(r'/home/mozzam/Documents/Few_patches/patches/R021A_reference_clean/edges2.jpg', edges2)

        # edges = cv2.Canny(img_blur, threshold, 10)
        # io.imshow(edges)
        # plt.imshow(edges)

        unique, counts = np.unique(edges1, return_counts=True)
        edge_to_image_ratio = counts[1]/counts[0]
        if edge_to_image_ratio < 0.01:
            io.imsave(r'/home/mozzam/Documents/Few_patches/patches/discarded/' + filename + '.jpg', img)
        else:
            io.imsave(r'/home/mozzam/Documents/Few_patches/patches/accepted/' + filename + '.jpg', img)

        # input_image = itk.imread(input_filename)
        # eager_normalized_image = itk.structure_preserving_color_normalization_filter(
        #     input_image,
        #     reference_image,
        #     color_index_suppressed_by_hematoxylin=0,
        #     color_index_suppressed_by_eosin=1,
        # )
        # dir_to_save = r'C:\Users\mozza\Documents\test\Few_patches\patches\M143A1_issues_normalised'
        # path_to_save = os.path.join(dir_to_save, filename)
        # itk.imwrite(eager_normalized_image, path_to_save)

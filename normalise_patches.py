import numpy as np
from skimage import io
import itk
import os
reference_image_filename = r'C:\Users\mozza\Documents\test\Few_patches\patches\R021A_reference_clean\1254.jpg'
reference_image = itk.imread(reference_image_filename)
#reference_reader = itk.ImageFileReader.New(FileName=reference_image_filename)


dir = r'C:\Users\mozza\Documents\test\Few_patches\patches\M143A1_issues'
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

    with open(os.path.join(dir, filename), 'r') as f:
        input_image = itk.imread(input_filename)
        eager_normalized_image = itk.structure_preserving_color_normalization_filter(
            input_image,
            reference_image,
            color_index_suppressed_by_hematoxylin=0,
            color_index_suppressed_by_eosin=1,
        )
        dir_to_save = r'C:\Users\mozza\Documents\test\Few_patches\patches\M143A1_issues_normalised'
        path_to_save = os.path.join(dir_to_save, filename)
        itk.imwrite(eager_normalized_image, path_to_save)

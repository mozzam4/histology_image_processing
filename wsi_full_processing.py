import tissueloc as tl
import numpy
import openslide
import os
from skimage import io, color
import numpy as np

# locate tissue contours with threshold 0.90 and rest as default
in_dir = r'/home/mozzam/Documents/Few_patches/selected_images'
out_dir = r'/home/mozzam/Documents/Few_patches/selected_images_cropped'
img_size = 4096
for filename in os.listdir(in_dir):
    print(filename)
    input_filename = os.path.join(in_dir, filename)

    cnts, d_factor = tl.locate_tissue_cnts(input_filename, max_img_size=img_size, smooth_sigma=13, thresh_val=0.90,
                                           min_tissue_size=10000)
    min_cnt = []
    max_cnt = []
    for cnt in cnts:
        cnt = cnt.squeeze(1)
        min_cnt.append(numpy.amin(cnt, axis=0))
        max_cnt.append(numpy.amax(cnt, axis=0))

    min_value = numpy.amin(numpy.array(min_cnt), axis=0)
    max_value = numpy.amax(numpy.array(max_cnt), axis=0)
    print(min_value)
    print(max_value)

    img = openslide.OpenSlide(input_filename)
    actual_initial_value = img.level_dimensions[0] * min_value/img_size
    actual_initial_value = actual_initial_value.astype(int)
    reqd_level = img.level_downsamples.index(d_factor)
    cropped_img = img.read_region(actual_initial_value, reqd_level, max_value - min_value)
    output_filename = os.path.join(out_dir, filename)
    cropped_img = np.asanyarray(cropped_img)
    cropped_img = color.rgba2rgb(cropped_img)
    output_filename = os.path.splitext(output_filename)[0] + '.jpg'
    io.imsave(output_filename, cropped_img)

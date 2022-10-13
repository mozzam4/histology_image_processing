# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt
#OPENSLIDE_PATH = r'C:\Users\mozza\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'
#os.environ['PATH'] = r'C:\Users\mozza\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin' + ';' + os.environ['PATH'] #can either specify openslide bin path in PATH, or add it dynamically
#os.add_dll_directory(OPENSLIDE_PATH)
#import openslide
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io, color
from sklearn import cluster
from colorthief import ColorThief
import itk
# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\Users\mozza\Downloads\openslide-win64-20220811\openslide-win64-20220811\bin'


if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def get_image_crop(img_path, img_coordinates):
    image = Image.open(img_path)
    return image.crop((img_coordinates['left'], img_coordinates['top'], img_coordinates['right'], img_coordinates['bottom']))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Taking the image crop
    path = r'C:\Users\mozza\Documents\test\B410D - 2021-03-15 11.53.57.ndpi'
    img = openslide.OpenSlide(path)
    reqd_level = img.level_count-4
    cropped_img = img.read_region((0, 0), reqd_level, img.level_dimensions[reqd_level])

    cropped_img.save(r'C:\Users\mozza\OneDrive\Desktop\test3.png')

    #Color Normalisation

    input_image_filename = r'C:\Users\mozza\Documents\test\png_format_test\new_img.png'
    reference_image_filename = r'C:\Users\mozza\Documents\test\R021A_reference_clean\2196.png'
    input_image_normalised = r'C:\Users\mozza\Documents\test\Normalised_images\normalised_test1.png'
    input_image = itk.imread(input_image_filename)
    reference_image = itk.imread(reference_image_filename)

    eager_normalized_image = itk.structure_preserving_color_normalization_filter(
        input_image,
        reference_image,
        color_index_suppressed_by_hematoxylin=0,
        color_index_suppressed_by_eosin=1,
    )
    itk.imwrite(eager_normalized_image, input_image_normalised)

    # # Color Identification
    #
    # img = ColorThief(r'C:\Users\mozza\Documents\test\Normalised_images\normalised_test7.png')
    # pallette = img.get_palette(color_count=10)
    # pallette = np.array(pallette)
    # len_1 = len(pallette)
    # print(pallette)
    # ind = np.linspace(0, len_1-1, len_1, dtype=int).reshape(1, len_1)
    # fig = plt.figure(figsize=(len_1, 2))
    # ax = fig.add_subplot(111)
    # ax.imshow(pallette[ind])
    # ax.set_yticks([])
    # plt.show()

    # Converting RGB images to LAB color space
    #color_img = io.imread(r'C:\Users\mozza\Documents\test\Normalised_images\normalised_test7.png')
    color_img = io.imread(r'C:\Users\mozza\Documents\test\M143A1_issues\ 0.jpg')
    #rgb_img = color.rgba2rgb(color_img)
    lab_img = color.rgb2lab(color_img)
    ab_image = lab_img[:, :, 1:3]
    X = ab_image.reshape((-1, 2)) # Only taking A and B component of LAB color space for clustering
    #X = color_img.reshape(-1, 4)
    km = cluster.k_means(X, n_clusters=5, n_init=4)
    #cluster = cluster.Birch(branching_factor=5280000, n_clusters=None).fit(X)
    cluster = cluster.DBSCAN().fit(X)
    print('ÄÄ')

    labels = cluster.labels_.flatten()
    centers = np.uint8(cluster.subcluster_centers_)
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(np.shape(color_img))
    #l_image = lab_img[:, :, 0:1] # Taking the L from LAB space of the image
    #labelled_image = cluster.labels_.reshape(np.shape(l_image))
    #lab_clustered = np.concatenate((l_image, segmented_image), axis=2) # Concatenating it with the  kmeans center
    #rgb_img = color.lab2rgb(lab_clustered)
    rgb_img = segmented_image
    rgb_img = np.asanyarray(rgb_img)

    io.imsave(r'C:\Users\mozza\Documents\test\Normalised_images\clustered_norm_7.png', rgb_img)
    plt.imshow(rgb_img)
    plt.imsave(r'C:\Users\mozza\Documents\test\Normalised_images\clustered_norm_7.png', rgb_img)
    #Image.fromarray(rgb_img, 'RGB').getcolors(20000)
    for each in np.unique(centers):
        rgb = cv.cvtColor(each, cv.COLOR_LAB2BGR)
        hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
        print(hsv)
# for each in centers:
#     #     c = b
#     #
#     #     c[:,:,0:1] = [100 if np.all(i == each) else 0 for i in c[:,:,1:3] for j in i]
#     #     # if c[:,:,1:3] != each:
#     #     #     c[:,:,0] = 100
#     #     # else:
#     #     #     c[:, :, 0] = 0
#     #
#     #     rgb_img = color.lab2rgb(c)
#     #     plt.imshow(rgb_img)
#
#
#     #Getting the canny edges
#     threshold = 100
#     img = cv.imread(r'C:\Users\mozza\OneDrive\Desktop\new_img_clean.png')
#     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     #img_gray[img_gray < threshold] = 0
#     #img_gray[img_gray >= threshold] = 255
#     img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
#     edges = cv.Canny(img_blur, 650, 60*3)
#     cv.imwrite(r'C:\Users\mozza\OneDrive\Desktop\normalised_edge_clean1.png', edges)
#
#     unique, counts = np.unique(edges, return_counts=True)
#     print(counts[1]/counts[0])
#
#     # Blur the image for better edge detection
#
#
#
#
#     print_hi('Mozzam')
#     path = r"D:\Scans fuer Michael\Scans R\R001A.ndpi"
#     coordinates = dict(left=5, top=5, right=50, bottom=50)
#     img_cropped = get_image_crop(path, coordinates)
#     img_cropped.show()
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

import matplotlib.image as img
import matplotlib.pyplot as plt
# Importing the modules
import pandas as pd
from scipy.cluster.vq import whiten
from sklearn import cluster
import numpy as np
import plotly.express as px

# Read image and print dimensions
image = img.imread(r'C:\Users\mozza\Documents\test\patches\R021A_reference_clean\ 2819.jpg')
print(image.shape)

# Store RGB values of all pixels in lists r, g and b
r = []
g = []
b = []

for row in image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)

# only printing the size of these lists
# as the content is too big
print(len(r))
print(len(g))
print(len(b))

# Saving as DataFrame
batman_df = pd.DataFrame({'red': r,
                          'green': g,
                          'blue': b})

# Scaling the values
batman_df['scaled_color_red'] = whiten(batman_df['red'])
batman_df['scaled_color_blue'] = whiten(batman_df['blue'])
batman_df['scaled_color_green'] = whiten(batman_df['green'])


# Preparing data to construct elbow plot.
# distortions = []
# num_clusters = range(1, 7)  # range of cluster sizes
#
# # Create a list of distortions from the kmeans function
# for i in num_clusters:
#     kmeans = cluster.k_means(batman_df[['scaled_color_red',
#                                         'scaled_color_blue',
#                                         'scaled_color_green']].to_numpy(), i)
#     cluster_centers = kmeans[0]
#     distortion = kmeans[2]
#     distortions.append(distortion)
#
# # Create a data frame with two lists, num_clusters and distortions
# elbow_plot = pd.DataFrame({'num_clusters': num_clusters,
#                            'distortions': distortions})
#
# # Create a line plot of num_clusters and distortions
# fig = px.line(elbow_plot, x='num_clusters', y='distortions')
# fig.show()

kmeans = cluster.k_means(batman_df[['scaled_color_red',
                                    'scaled_color_blue',
                                    'scaled_color_green']], 4)

cluster_centers = kmeans[0]
dominant_colors = []

# Get standard deviations of each color
red_std, green_std, blue_std = batman_df[['red',
                                          'green',
                                          'blue']].std()

for cluster_center in cluster_centers:
    red_scaled, green_scaled, blue_scaled = cluster_center

    # Convert each standardized value to scaled value
    dominant_colors.append((
        red_scaled * red_std / 255,
        green_scaled * green_std / 255,
        blue_scaled * blue_std / 255
    ))

# Display colors of cluster centers
plt.imshow([dominant_colors])
plt.show()
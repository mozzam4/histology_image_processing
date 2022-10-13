import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
import random
with open(r'C:\Users\mozza\Documents\test\M200A2 - 2021-03-15 13.07.52.json', 'r') as final:
    patches_colors = json.load(final)

red = []
green = []
blue = []
colors = []
for patch in patches_colors:
    for each in patch['pallette']:
        red.append(each[0])
        green.append(each[1])
        blue.append(each[2])
        colors.append(each)
# red = np.array(red) / 255
# green = np.array(green) / 255
# blue = np.array(blue) /255
colors = np.array(colors) / 255
#colors = random.sample(colors, k=2000)
#AP_model = cluster.AffinityPropagation(damping=0.9)
optics_model = cluster.OPTICS(eps=0.75, min_samples=10)
optics_result = optics_model.fit_predict(colors)
#mean_model = cluster.MeanShift()
#clusters = mean_model.fit_predict(colors)

#cluster = cluster.DBSCAN().fit(colors)

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)

ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")

ax.scatter(red, green, blue, marker='o', s=100, facecolors=colors)
plt.show()
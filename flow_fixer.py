import os
import numpy as np
from Vector_Extractor import img2rgb
from PIL import Image
import matplotlib.pyplot as plt

#fix a small error from generating flow files in another script


def viewFlow(flow_array, dimx, dimy):
    colormap = img2rgb(flow_array, dimx, dimy)
    im = Image.fromarray(colormap)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

data_path = "C://data//FlyingMonkeys_1//test"

file_names = sorted(os.listdir(data_path))

i = 0
for file in file_names:
    if file.startswith("gt"):
        flow = np.load(os.path.join(data_path, file))
        print(np.shape(flow))
        #flow[:, 0:8,:] = 0
        #flow[:, -8:, :] = 0
        viewFlow(flow, 256, 192)
        #np.save(os.path.join(data_path, file), flow)
        i = i + 1
        if i > 5:
            break



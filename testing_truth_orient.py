import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from Vector_Extractor import exr2flow, img2rgb

data_path = "F://Flying_monkeys_2_RGB//test"

file_names = sorted(os.listdir(data_path))
i = 0

visual = False

for file in file_names:
    if file.startswith("gt"):

        i = i + 1
        #img = np.load(os.path.join(data_path, file))
        flow = np.load(os.path.join(data_path, file))

        flow = np.reshape(flow,(192,256,2))
        #print(np.shape(flow))

        print(np.shape(flow))

        if visual:
            truth_colormap = img2rgb(flow, 256, 192)
            im = Image.fromarray(truth_colormap)
            fig, ax = plt.subplots(1)
            ax.imshow(im)

        np.save(os.path.join(data_path, file), flow)
        #if i > 10:
        #    break
        print(i)
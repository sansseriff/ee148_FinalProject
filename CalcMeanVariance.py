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

data_path = "F://Flying_monkeys_2_RGB//train"

file_names = sorted(os.listdir(data_path))

meanBoolean = 0
meanTime = 0
stgBoolean = 0
stgTime = 0

i = 0
for file in file_names:
    if file.startswith("gt"):
        i = i + 1
        #img = np.load(os.path.join(data_path, file))
        flow = np.load(os.path.join(data_path, file))
        #print(np.shape(flow))

        print("axis 1: ",flow[75:85, 150:160, 0])
        print("axis 2: ", flow[75:85, 150:160, 1])

        if False:
            meanBoolean = meanBoolean + np.mean(img[:,:,0])
            meanTime = meanTime + np.mean(img[:, :, 1])
            stgBoolean = stgBoolean + np.std(img[:,:,0])
            stgTime = stgTime + np.std(img[:, :, 1])


        if i > 10:
            break
if False:
    meanBoolean = meanBoolean/i
    meanTime = meanTime/i
    stgBoolean = stgBoolean/i
    stgTime = stgTime/i

    print("meanBoolean:", meanBoolean)
    print("meanTime:", meanTime)
    print("stgBoolean:", stgBoolean)
    print("stgTime:", stgTime)
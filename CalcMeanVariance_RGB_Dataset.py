import os
import numpy as np
from Vector_Extractor import img2rgb
from PIL import Image
import matplotlib.pyplot as plt

#calc mean and variance of RGB dataset


def viewFlow(flow_array, dimx, dimy):
    colormap = img2rgb(flow_array, dimx, dimy)
    im = Image.fromarray(colormap)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

data_path = "F://Flying_monkeys_2_RGB//train"

file_names = sorted(os.listdir(data_path))

meanBooleanX = 0
meanTimeX = 0
stgBooleanX = 0
stgTimeX = 0

meanBooleanY = 0
meanTimeY = 0
stgBooleanY = 0
stgTimeY = 0

meanBooleanZ = 0
meanTimeZ = 0
stgBooleanZ = 0
stgTimeZ = 0

i = 0
for file in file_names:
    if file.startswith("data"):
        i = i + 1
        img = np.load(os.path.join(data_path, file))
        #flow = np.load(os.path.join(data_path, file))
        #print(np.shape(flow))

        #print("axis 1: ",flow[75:85, 150:160, 0])
        #print("axis 2: ", flow[75:85, 150:160, 1])

        if True:
            meanBooleanX = meanBooleanX + np.mean(img[:,:,0])
            meanTimeX = meanTimeX + np.mean(img[:, :, 1])
            stgBooleanX = stgBooleanX + np.std(img[:,:,0])
            stgTimeX = stgTimeX + np.std(img[:, :, 1])

            meanBooleanY = meanBooleanY + np.mean(img[:, :, 2])
            meanTimeY = meanTimeY + np.mean(img[:, :, 3])
            stgBooleanY = stgBooleanY + np.std(img[:, :, 2])
            stgTimeY = stgTimeY + np.std(img[:, :, 3])

            meanBooleanZ = meanBooleanZ + np.mean(img[:, :, 4])
            meanTimeZ = meanTimeZ + np.mean(img[:, :, 5])
            stgBooleanZ = stgBooleanZ + np.std(img[:, :, 4])
            stgTimeZ = stgTimeZ + np.std(img[:, :, 5])


        #if i > 10:
         #   break
if True:


    meanBooleanX = meanBooleanX / i
    meanBooleanY = meanBooleanY / i
    meanBooleanZ = meanBooleanZ / i

    meanTimeX = meanTimeX/i
    meanTimeY = meanTimeY / i
    meanTimeZ = meanTimeZ / i

    stgBooleanX = stgBooleanX/i
    stgBooleanY = stgBooleanY / i
    stgBooleanZ = stgBooleanZ / i


    stgTimeX = stgTimeX/i
    stgTimeY = stgTimeY / i
    stgTimeZ = stgTimeZ / i

    print(i)
    MeanNorm = [meanBooleanX,meanTimeX, meanBooleanY,meanTimeY,meanBooleanZ,meanTimeZ]
    StgNorm = [stgBooleanX, stgTimeX, stgBooleanY, stgTimeY, stgBooleanZ, stgTimeZ]

    print(MeanNorm)
    print(StgNorm)
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Vector_Extractor import exr2flow, img2rgb
import cv2
import statistics

image_gen = False


data_path = 'F:\data\proof_data'
dataset_path = 'C:\data\dataset'
info_path = 'C:\data\ex'
file_names = sorted(os.listdir(data_path))
#seperate_names = [[0,0,0] for i in range(len(file_names))]
examples = [0 for i in range(len(file_names))]

for i, name in enumerate(file_names):
        #print(name.split('_'))
        seperate_names = name.split('_')
        examples[i] = int(seperate_names[0][2:])

min_example = min(examples)
max_example = max(examples)


'''
q = np.array([0,0,0,0])
a = np.array([4,432,2,6])
for i in range(100):
    q = q + np.random.poisson(a*.1)

q = q/100.3
print(q)
'''
#example_set = np.zeros((max_example - min_example,200,240,2))

i = 999
photons_list = []
lost_photons_list = []

#for ex in range(1, 50):
for ex in range(0, max_example - min_example):
    lost_photons = 0
    sump = 0
    i = i + 1
    #example_set = (np.zeros((200,240,2)),np.zeros((200,240,2))) #tuple of image arrays each with two channels per pixel

    example = np.zeros((200,240,2))
    truth = np.zeros((200,240,2))
    #example is a 3d array with length and height of original image,
    # and 2 channels for each pixel
    #example = example_set[ex]
    Img_base = 'ex' + str(ex + min_example) + '_Img_00'
    Flow_base = 'ex' + str(ex + min_example) + '_Vector_00'

    for timeslot in range(64):
        file_name_img = Img_base + str(timeslot).zfill(2) + '.tif'
        file_name_flow = Flow_base + str(timeslot).zfill(2) + '.exr'
        #print(file_name)
        slot = Image.open(os.path.join(data_path, file_name_img))
        slot = slot.convert('L')
        I = np.asarray(slot)
        points = np.zeros(np.shape(I))
        points = np.random.poisson(I*.0005)


        example[:,:,0] = example[:,:,0] + points

        # first channel for each pixel is a boolean
        # set to 1 if 1 or more photons hit the pixel during the example, zero otherwise
        example[:, :, 0][example[:, :, 0] > 1] = 1


        sump = sump + np.sum(points)
        # lost photons are 2nd, 3rd, 4th etc. photons that hit the same pixel. After the first photon,
        # most SPAD or SNSPD arrays have a fairly long dead-time when they are not sensitive


        #if timeslot == 34:

        #    print(example[100:110, 120:130, 0])
        #    print()
        #    print()
        #    print(points[100:110, 120:130])
        #    print("sump: ", sump)
        #    print("sum of floored: ", np.sum(example[:, :, 0]))
        #    print("lost photons:", lost_photons)




        # 2nd channel for each pixel is the arrival time of a photon
        # for now the time is discretized into 64 time bins (corresponding to the 64 rendered
        # images for each example)
        example[:, :, 1][(example[:, :, 1] < 1) & (example[:, :, 0] > 0)] = timeslot




        #imslot = Image.fromarray(np.uint8(points*255))
        #imslot.save(os.path.join(info_path, str(ex) + '_slot_' + str(timeslot) + '.png'))

        #image_set = exr2flow(os.path.join(data_path,file_name_flow ), 240, 200)
        #flow_image = cv2.cvtColor(image_set[1], cv2.COLOR_BGR2RGB)


        #fig, ax = plt.subplots(1)
        #ax.imshow(flow_image)

        #flow_Image = Image.fromarray(np.uint8(flow_image))
        #flow_Image.save(os.path.join(info_path, str(ex) + '_slot_' + str(timeslot) + '.png'))


        #ax.imshow(imslot, cmap='gray')

        #points[points > 1] = 1

        #print(I[75:76])


    lost_photons = sump - np.sum(example[:, :, 0])


    file_name_flow = Flow_base + str(32).zfill(2) + '.exr'
    # print(file_name)
    image_set = exr2flow(os.path.join(data_path,file_name_flow ), 240, 200)


    truth = image_set[0]
    example_set = (example, truth)

    if image_gen:
        truth_colormap = img2rgb(example_set[1] ,240, 200)
        im = Image.fromarray(truth_colormap)
        fig, ax = plt.subplots(1)
        ax.imshow(im)
    ex_file_name = 'ex_' + str(i)
    np.save(os.path.join(dataset_path, ex_file_name ), example_set)

    #print(sump)
    #print(lost_photons)
    #print()
    photons_list.append(sump)
    lost_photons_list.append(lost_photons)

print("median of lost photons", statistics.median(lost_photons_list))
print("median of photons", statistics.median(photons_list))

plt.figure()
plt.hist(photons_list, bins=100)  # arguments are passed to np.histogram
plt.title("Histogram of photon detections for each example")

plt.figure()
plt.hist(lost_photons_list, bins=100)  # arguments are passed to np.histogram
plt.title("Histogram of lost photon for each example")
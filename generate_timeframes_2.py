import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Vector_Extractor import exr2flow, img2rgb
import cv2
import statistics
from PIL import ImageEnhance

image_gen = False
visualsave = False


print("running")
data_path = 'C://data//proof_data_2'
dataset_train_path = 'F://Flying_monkeys_2_RGB//train'
dataset_test_path = 'F://Flying_monkeys_2_RGB//test'
dataset_path = 'C://data//FlyingMonkeys_2'

ts = 0
tr = 0

info_path = "F://info_11"
file_names = sorted(os.listdir(data_path))
#seperate_names = [[0,0,0] for i in range(len(file_names))]
examples = [0 for i in range(len(file_names))]

for i, name in enumerate(file_names):
        #print(name.split('_'))
        seperate_names = name.split('_')
        examples[i] = int(seperate_names[0][2:])

        if i % 10000 == 0:
            print("10000")

min_example = min(examples)
max_example = max(examples)

################ 75% is in train set ###############
number_train = int(0.75*(max_example - min_example))
print("number train: ", number_train)

'''
q = np.array([0,0,0,0])
a = np.array([4,432,2,6])
for i in range(100):
    q = q + np.random.poisson(a*.1)

q = q/100.3
print(q)
'''
#example_set = np.zeros((max_example - min_example,192,256,2))

i = 999
photons_list = []
lost_photons_list = []

#for ex in range(1, 50):
for ex in range(max_example - min_example):
#for ex in range(9,10):
    print("cycle ", ex)
    lost_photons = 0
    sump = 0
    i = i + 1
    #example_set = (np.zeros((192,256,2)),np.zeros((192,256,2))) #tuple of image arrays each with two channels per pixel

    example = np.zeros((192,256,6))
    truth = np.zeros((192,256,2))
    flow_Image = np.zeros((192, 256, 3))
    points_Image = np.zeros((192, 256, 3))
    #example is a 3d array with length and height of original image,
    # and 2 channels for each pixel
    #example = example_set[ex]
    Img_base = 'ex' + str(ex + min_example) + '_Img_0'
    Flow_base = 'ex' + str(ex + min_example) + '_Vector_0'

    for timeslot in range(128):
        file_name_img = Img_base + str(timeslot).zfill(3) + '.tif'
        file_name_flow = Flow_base + str(timeslot).zfill(3) + '.exr'
        #print(file_name)


        #load and increase saturation
        img = Image.open(os.path.join(data_path, file_name_img))
        converter = ImageEnhance.Color(img)
        slot = converter.enhance(2.0)


        #slot = slot.convert('L')
        #4th channel is alpha channel. not used
        I = np.asarray(slot)[:,:,:3] #disregard alpha

        if False:
            print(np.shape(I))
            print(I[5:10,5:10,0])
            print('###################')
            print(I[5:10, 5:10,1])
            print('###################')
            print(I[5:10, 5:10,2])
            print('###################')
            print(I[5:10, 5:10,3])
            print('###################')
            print('###################')


        points = np.zeros((192,256,3))
        points = np.random.poisson(I*.00025)  #halved from the previous dataset because the frames are doubled
        #print()


        #layout of tensor channels
        #[0, 1, 2, 3, 4, 5]
        #[B, T, B, T, B, T]


        # first channel for each pixel is a boolean
        # set to 1 if 1 or more photons hit the pixel during the example, zero otherwise
        # using numpy boolean mask notation
        example[:,:,0] = example[:,:,0] + points[:,:,0]
        example[:, :, 0][example[:, :, 0] > 1] = 1

        example[:, :, 2] = example[:, :, 2] + points[:, :, 1]
        example[:, :, 2][example[:, :, 2] > 1] = 1

        example[:, :, 4] = example[:, :, 4] + points[:, :, 2]
        example[:, :, 4][example[:, :, 4] > 1] = 1

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
        example[:, :, 3][(example[:, :, 3] < 1) & (example[:, :, 2] > 0)] = timeslot
        example[:, :, 5][(example[:, :, 5] < 1) & (example[:, :, 4] > 0)] = timeslot







        if visualsave:
            points8 = np.uint8(points * 255)
            if False:
                print(np.shape(points8))
                print(points8[15:25, 15:25, 0])
                print('###################')
                print(points8[15:25, 15:25, 1])
                print('###################')
                print(points8[15:25, 15:25, 2])
                print('###################')
                print('###################')
            imslot = Image.fromarray(np.uint8(points * 255))
            imslot.save(os.path.join(info_path, str(ex) + '_slot_' + str(timeslot) + '.png'))

            if timeslot == 64:
                image_set = exr2flow(os.path.join(data_path, file_name_flow), 256, 192)
                flow_image = cv2.cvtColor(image_set[1], cv2.COLOR_BGR2RGB)
                flow_Image = Image.fromarray(np.uint8(flow_image))
                flow_Image.save(os.path.join(info_path, str(ex) + '_slot_flow_' + str(timeslot) + '.png'))
            #image_set = exr2flow(os.path.join(data_path,file_name_flow ), 256, 200)
            #flow_image = cv2.cvtColor(image_set[1], cv2.COLOR_BGR2RGB)


        #fig, ax = plt.subplots(1)
        #ax.imshow(flow_image)

        #flow_Image = Image.fromarray(np.uint8(flow_image))
        #flow_Image.save(os.path.join(info_path, str(ex) + '_slot_' + str(timeslot) + '.png'))


        #ax.imshow(imslot, cmap='gray')

        #points[points > 1] = 1

        #print(I[75:76])


    lost_photons = sump - np.sum(example[:, :, 0])


    file_name_flow = Flow_base + str(timeslot).zfill(3) + '.exr'
    # print(file_name)
    image_set = exr2flow(os.path.join(data_path,file_name_flow ), 192, 256) ####################################################


    truth = image_set[0]

    if visualsave:
        points_Image[:, :, 0] = example[:, :, 1]
        points_Image[:, :, 1] = example[:, :, 3]
        points_Image[:, :, 2] = example[:, :, 5]
        pointsIMG = Image.fromarray(np.uint8(points_Image[:,:,0] * 255*2))
        pointsIMG.save(os.path.join(info_path, str(ex) + 'R.png'))
        pointsIMG = Image.fromarray(np.uint8(points_Image[:, :, 1] * 255 * 2))
        pointsIMG.save(os.path.join(info_path, str(ex) + 'G.png'))
        pointsIMG = Image.fromarray(np.uint8(points_Image[:, :, 2] * 255 * 2))
        pointsIMG.save(os.path.join(info_path, str(ex) + 'B.png'))


    example_set = (example, truth)

    if image_gen:
        truth_colormap = img2rgb(example_set[1] ,192, 256)
        im = Image.fromarray(truth_colormap)
        fig, ax = plt.subplots(1)
        ax.imshow(im)
    #ex_file_name = 'ex_' + str(i)
    #np.save(os.path.join(dataset_path, ex_file_name ), example_set)

    #print(sump)
    #print(lost_photons)
    #print()
    photons_list.append(sump)
    lost_photons_list.append(lost_photons)

    if ex < number_train:
        traindata_filename = "data_" + str(tr)
        traingt_filename = "gt_" + str(tr)
        np.save(os.path.join(dataset_train_path, traindata_filename), example)
        np.save(os.path.join(dataset_train_path, traingt_filename), truth)
        tr = tr + 1

    else:
        traindata_filename = "data_" + str(ts)
        traingt_filename = "gt_" + str(ts)
        np.save(os.path.join(dataset_test_path, traindata_filename), example)
        np.save(os.path.join(dataset_test_path, traingt_filename), truth)
        ts = ts + 1



print("median of lost photons", statistics.median(lost_photons_list))
print("median of photons", statistics.median(photons_list))

plt.figure()
plt.hist(photons_list, bins=100)  # arguments are passed to np.histogram
plt.title("Histogram of photon detections for each example")

plt.figure()
plt.hist(lost_photons_list, bins=100)  # arguments are passed to np.histogram
plt.title("Histogram of lost photon for each example")
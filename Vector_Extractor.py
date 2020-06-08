########################################

#modified from code by Tobias Weis
# see: http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/

########################################

import array
import OpenEXR
import Imath
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import csv
import time
import datetime
import h5py

def exr2flow(exr, w,h):
  file = OpenEXR.InputFile(exr)

  # Compute the size
  dw = file.header()['dataWindow']
  sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

  FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
  (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

  img = np.zeros((h,w,2), np.float64)
  img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
  img[:,:,1] = -np.array(G).reshape(img.shape[0],-1)

  hsv = np.zeros((h,w,3), np.uint8)
  hsv[...,1] = 255

  mag, ang = cv2.cartToPolar(img[...,0], img[...,1])
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

  return img, bgr, mag,ang


def exr2numpy(exr, maxvalue=1., normalize=True):
  """ modified from code by Tobias Weis"""
  """ converts 1-channel exr-data to 2D numpy arrays """
  file = OpenEXR.InputFile(exr)

  # Compute the size
  dw = file.header()['dataWindow']
  sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

  # Read the three color channels as 32-bit floats
  FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
  (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R")]

  # create numpy 2D-array
  img = np.zeros((sz[1], sz[0], 3), np.float64)

  # normalize
  data = np.array(R)
  data[data > maxvalue] = maxvalue

  if normalize:
    data /= np.max(data)

  img = np.array(data).reshape(img.shape[0], -1)

  return img




def img2rgb(img,w,h):

  #####what is going on???

  hsv = np.zeros((h, w, 3), np.uint8)
  # hsv = np.zeros((h, w, 3), np.uint8)

  hsv[..., 1] = 255

  mag, ang = cv2.cartToPolar(img[..., 0], img[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  return rgb


'''

for i,j,y in os.walk('..'):
    print(i)

entries = os.listdir('..')
print(entries)

#testing_path = '../data/proof_data'
testing_path = 'C:\data\proof_data'
image_set = exr2flow(os.path.join(testing_path, 'the_exr_thing1402.exr'),240 ,240 )

#data = exr2numpy(os.path.join(testing_path, 'Speed0003.exr'), normalize=False)


#fig = plt.figure()
#plt.imshow(data)
#plt.colorbar()
#plt.show()

print(len(image_set[1]))
print(type(image_set))

image = image_set[3]
image = cv2.cvtColor(image_set[1], cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

'''
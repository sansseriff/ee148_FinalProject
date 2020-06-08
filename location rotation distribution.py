import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
import numpy as np
import os

font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 16}



path = "..//blender_data"
locations = np.load(os.path.join(path,"locations.npy"))
rotations = np.load(os.path.join(path,"rotations.npy"))
shape = np.shape(locations)
delta_shape = (3815,3)

delta_loc = np.zeros(delta_shape)
delta_loc = (locations[:3814,0,:] - locations[:3814,1,:])*100

delta_rot = np.zeros(delta_shape)
delta_rot = (rotations[:3814,0,:] - rotations[:3814,1,:])*128*(180/3.14159)

'''
for item in range(len(locations)):
    if locations[item,0,0] == 0 and\
        locations[item,0,1] == 0 and\
        locations[item,0,2] == 0 and\
        locations[item,1,0] == 0:
        print(item)
        break
'''
matplotlib.rc('font', **font)
print(np.min(delta_rot))
print(np.max(delta_loc))

fig, axs = plt.subplots(1,2)
fig.tight_layout()
axs[0].hexbin(delta_rot[:,0],delta_rot[:,1])
axs[0].set(xlabel='y-label', ylabel='y-label')
axs[1].set(xlabel='y-label', ylabel='y-label')
axs[1].set(title='Velocity in XY Plane')
axs[0].set(title='Rotational Velocity Along X and Y Axes')

A = np.histogramdd(delta_rot[:,0:2],bins = 100)




axs[1].hexbin(delta_loc[:,0],delta_loc[:,1])
A = np.histogramdd(delta_rot[:,0:2],bins = 100)




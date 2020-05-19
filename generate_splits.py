import numpy as np
import os
import json


dataset_path = 'C:\data\dataset'
file_names = sorted(os.listdir(data_path))
#seperate_names = [[0,0,0] for i in range(len(file_names))]
examples = [0 for i in range(len(file_names))]

for i, name in enumerate(file_names):
        #print(name.split('_'))
        seperate_names = name.split('_')
        examples[i] = int(seperate_names[0][2:])

min_example = min(examples)
max_example = max(examples)

for example in examples:
    np.open(file_names)
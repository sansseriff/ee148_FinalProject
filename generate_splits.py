import numpy as np
import os
import json




dataset_path = 'C:\data\dataset'
splits_path = 'C:\data\splits'
file_names = sorted(os.listdir(dataset_path))
#seperate_names = [[0,0,0] for i in range(len(file_names))]
examples = [0 for i in range(0,len(file_names))]

for i, name in enumerate(file_names):
        #print(name.split('_'))
        seperate_names = name.split('_')
        seperate_names = seperate_names[1].split('.')
        examples[i] = int(seperate_names[0])
        #examples[i] = int(seperate_names[0][2:])


#sort or randomize list loading here:
examples.sort()





number_train = int(0.75*len(examples))
number_test = int(len(examples) - number_train)
train_data = np.zeros((number_train,192,256,2))
test_data = np.zeros((number_test,192,256,2))

train_gt = np.zeros((number_train,192,256,2))
test_gt = np.zeros((number_test,192,256,2))


min_example = min(examples)
max_example = max(examples)

#filename = "ex_1.npy"
#ex1 = np.load(os.path.join(dataset_path, filename))

train_idx = 0
test_idx = 0
for i, example in enumerate(examples):
        file_name = 'ex_' + str(example) + '.npy'
        ex = np.load(os.path.join(dataset_path, file_name))
        if i < number_train:
                #ex_c = ex[0][4:-4]


                #ex_c = np.pad(ex[0][4:-4], ((0,0), (8,8), (0,0)), 'minimum')
                #print(np.shape(ex_c))

                #this is used to bring resolution to 256 x 192
                train_data[train_idx] = np.pad(ex[0][4:-4], ((0,0), (8,8), (0,0)), 'minimum')
                train_gt[train_idx] = np.pad(ex[1][4:-4], ((0,0), (8,8), (0,0)), 'minimum')
                train_idx = train_idx + 1

        else:

                test_data[test_idx] = np.pad(ex[0][4:-4], ((0,0), (8,8), (0,0)), 'minimum')
                test_gt[test_idx] = np.pad(ex[1][4:-4], ((0,0), (8,8), (0,0)), 'minimum')
                test_idx = test_idx + 1
        if i % 100 == 0:
                print("one hundred")

np.shape(train_data)
np.shape(train_gt)
np.shape(test_data)
np.shape(test_gt)


np.save(os.path.join(splits_path, "train_data"), train_data)
np.save(os.path.join(splits_path, "train_gt"), train_gt)

np.save(os.path.join(splits_path, "test_data"), test_data)
np.save(os.path.join(splits_path, "test_gt"), test_gt)
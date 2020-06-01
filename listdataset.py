import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np



def default_loader(root, path_data, path_gt):

    data = np.load(os.path.join(root, path_data))
    gt = np.load(os.path.join(root, path_gt))
    # return 2 numpy arrays
    return data, gt

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):               #acctually loads the file when this is called
        input, target = self.path_list[index]

        #input and target change from lists to numpy arrays. That's weird...
        input, target = self.loader(self.root, input, target)
        if self.co_transform is not None:
            inputs, target = self.co_transform(input, target)
        if self.transform is not None:
            input[0] = self.transform(input[0])
            input[1] = self.transform(input[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.path_list)

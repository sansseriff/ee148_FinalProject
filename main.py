from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from PhotonNetS import PhotonNetS
from Vector_Extractor import img2rgb

import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
import pickle
from PIL import Image




import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sn
import pandas as pd
from math import log
import math

from listdataset import ListDataset


random.seed(2020)
torch.manual_seed(2020)


import os


def viewFlow(flow_array, dimx, dimy):
    colormap = img2rgb(flow_array, dimx, dimy)
    im = Image.fromarray(colormap)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

def viewImg(img):
    im = Image.fromarray(img)
    fig, ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output, hidden_layer = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

        total_loss = total_loss + loss.item()
    #train loss for each epoch is an average of the loss over all mini-batches
    train_loss = total_loss/batch_idx

    return train_loss


def test(model, device, test_loader, evaluate = False):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0

    images = []
    allimages = []
    master_preds = []
    master_truths = []
    master_hidden_layers = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hidden_layer = model(data)

            #feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            print(len(hidden_layer))
            print(len(hidden_layer[0]))
            #print(hidden_layer[0])


            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)


            if evaluate:
                for i in range(len(pred)):
                    master_preds.append(pred[i][0].item())
                    master_truths.append(target[i].item())
                    layer = hidden_layer[i].cpu()
                    master_hidden_layers.append(layer.numpy())
                    image = data[i][0].cpu()
                    allimages.append(image.numpy())
                    if pred[i][0] == target[i]:
                        continue
                    else:
                        #print("not equal")
                        #print("pred is ", pred[i][0].item(), "and target is ", target[i].item())
                        image = data[i][0].cpu()
                        images.append([image.numpy(),pred[i][0].item(),target[i].item()])

        if evaluate:

            #print(len(master_hidden_layers))
            #print(master_hidden_layers[0])

            distances = np.zeros(len(master_hidden_layers))

            #x0 = master_hidden_layers[0]

            for i in range(len(distances)):
                length = 0
                for dim in range(len(master_hidden_layers[0])):
                    length = length + (master_hidden_layers[i][dim] - master_hidden_layers[15][dim])**2
                length = math.sqrt(length)
                distances[i] = length

            sorted_distance_index = np.argsort(distances)

            figa = plt.figure()


            print("test")
            for i in range(9):
                sub = figa.add_subplot(9, 1, i + 1)
                sub.imshow(allimages[sorted_distance_index[i]], interpolation='nearest', cmap='gray')

            X = master_hidden_layers
            y = np.array(master_truths)
            tsne = TSNE(n_components=2, random_state=0)
            X_2d = np.array(tsne.fit_transform(X))

            target_ids = range(10)

            cdict = {0: 'orange', 1: 'red', 2: 'blue', 3: 'green', 4: 'salmon', 5:'c', 6: 'm', 7: 'y', 8: 'k', 9: 'lime'}

            fig, ax = plt.subplots()
            for g in np.unique(y):
                ix = np.where(y == g)
                ax.scatter(X_2d[ix, 0], X_2d[ix, 1], c=cdict[g], label=g, s=5)
            ax.legend()
            plt.show()


            #i = 1
            #plt.figure(figsize=(6, 5))
            #plt.scatter(X_2d[10*i:10*i+10,0],X_2d[:10,1])



            CM = confusion_matrix(master_truths,master_preds)
            CMex = CM
            #for i in range(len(CM)):
            #    for j in range(len(CM)):
            #        if CM[i][j] > 0:
            #            CMex[i][j] = log(CM[i][j])
            #        else:
            #            CMex[i][j] = CM[i][j]

            print(CM)
            print(CMex)

            df_cm = pd.DataFrame(CM, range(10), range(10))
            #plt.figure(figsize=(10,7))
            fig0,ax0 = plt.subplots(1)
            sn.set(font_scale=1)  # for label size
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 11})  # font size
            #ax0.set_ylim(len(CMex) - 0.5, 0.5)
            plt.xlabel("predicted")
            plt.ylabel("ground truth")
            plt.show()




            fig = plt.figure()

            for i in range(9):
                sub = fig.add_subplot(3, 3, i + 1)
                sub.imshow(images[i + 10][0], interpolation='nearest', cmap='gray')

                title = "Predicted: " + str(images[i+ 10][1]) + " True: " + str(images[i+ 10][2])
                sub.set_title(title)

            kernels = model.conv1.weight.cpu().detach().clone()
            kernels = kernels - kernels.min()
            kernels = kernels / kernels.max()

            kernels = kernels.numpy()
            print(np.shape(kernels))

            fig2 = plt.figure()
            for i in range(8):

                sub = fig2.add_subplot(2, 4, i + 1)
                sub.imshow(kernels[i][0], interpolation='nearest', cmap='gray')

                title = "Kernel #" + str(i + 1)
                sub.set_title(title)


        #fig, axs = plt.subplots(3, 3, constrained_layout=True)
        #for i in range(9):
        #    fig[i].imshow(images[i][0], interpolation='nearest', cmap='gray')
        #    axs[i].set_title("all titles")





    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch PhotonNet')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:

        '''
        set this up like the training set!
        '''
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader, evaluate = True)

        return

    # this takes too long
    '''
    datasets_path = "C:\data\splits"
    test_data = np.load(os.path.join(datasets_path, "test_data.npy"))
    test_gt = np.load(os.path.join(datasets_path, "test_gt.npy"))
    train_data = np.load(os.path.join(datasets_path, "train_data.npy"))
    train_gt = np.load(os.path.join(datasets_path, "train_gt.npy"))

    print(np.shape(test_data))

    '''
    FM_1_testdir = "C://data//FlyingMonkeys_1//test"
    FM_1_traindir = "C://data//FlyingMonkeys_1//train"
    train_data = []
    train_gt = []

    test_data = []
    test_gt = []

    for filename in os.listdir(FM_1_traindir):
        if filename.startswith("data"):
            #this is so hack-y...
            train_data.append([filename, int(filename.split('_')[1].split('.')[0])])  #split the filename
            # by the underscore and split the number from the filetype
        elif filename.startswith("gt"):
            train_gt.append([filename, int(filename.split('_')[1].split('.')[0])])

    for filename in os.listdir(FM_1_testdir):
        if filename.startswith("data"):
            #this is so hack-y...
            test_data.append([filename, int(filename.split('_')[1].split('.')[0])])  #split the filename
            # by the underscore and split the number from the filetype
        elif filename.startswith("gt"):
            test_gt.append([filename, int(filename.split('_')[1].split('.')[0])])

    train_data.sort(key = lambda x: x[1])
    train_gt.sort(key=lambda x: x[1])
    test_data.sort(key=lambda x: x[1])
    test_gt.sort(key=lambda x: x[1])

    train_list = [[data[0],gt[0]] for data, gt in zip(train_data,train_gt)]
    test_list = [[data[0], gt[0]] for data, gt in zip(test_data, test_gt)]

    print(train_list [787])
    print(test_list[787])
    train_dataset = ListDataset(FM_1_traindir , train_list)
    test_dataset = ListDataset(FM_1_testdir, test_list)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers = 1, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers = 1, pin_memory=True, shuffle=False)

    '''
    a,b = train_dataset.__getitem__(178)

    print(np.shape(a))
    viewFlow(b,256,192)

    np.shape(a)
    print(a[: ,: ,0].dtype)
    print(a[:, :, 1].dtype)
    viewImg(a[:,:,1])
    viewImg(a[:, :, 0])
    '''



    '''
    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
    
    train_dataset_augmented = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([  # Data preprocessing
                                       #transforms.RandomCrop(28, padding=(1, 1, 1, 1)),
                                       #transforms.RandomRotation(4, resample=PIL.Image.BILINEAR),
                                       #transforms.RandomResizedCrop(28, scale=(0.85, 1.0), ratio=(1, 1),
                                       #                             interpolation=2),
                                       transforms.RandomAffine(8, translate=(.065, .065), scale=(0.80, 1.1),
                                                               resample=PIL.Image.BILINEAR),
                                       transforms.ToTensor(),  # Add data augmentation here
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    
    print(type(train_dataset))
    print(len(train_dataset), type(train_dataset[0][0]), type(train_dataset[0][1]), type(train_dataset[0]))

    print("the int is: ", train_dataset[2][1])
    print(np.shape(train_dataset[0][0][0].numpy()))

    idx = [[] for i in range(10)]
    #each row of indexes is a list of indexes in the train_dataset
    #e.g. row 5 containes a list of indexes for the places in train_dataset with images of 5
    print(idx[4])
    for i, img in enumerate(train_dataset):

        #if False:
        if i < 5:
            fig = plt.figure()
            plt.imshow(img[0][0].numpy(), cmap='gray')

            fig = plt.figure()
            plt.imshow(train_dataset_augmented[i][0][0].numpy(), cmap='gray')

        for number in range(10):
            if img[1] == number:
                idx[number].append(i)


    val_idx = [[] for i in range(10)]
    train_idx = [[] for i in range(10)]
    #print(idx[0][1:100])

    for i, number_indx in enumerate(idx):
        random.shuffle(number_indx)     #shuffle the index list
        l = len(number_indx)
        idx_lim = int(l*0.15)           # 15% of train set is used for validation
        val_idx[i] = number_indx[0:idx_lim]
        train_idx[i] = number_indx[idx_lim:]


    subset_indices_train = [j for sub in train_idx for j in sub]
    subset_indices_valid = [j for sub in val_idx for j in sub]


    # for adjusting size of train set

    train_length = int(len(subset_indices_train))
    #train_length = int(len(subset_indices_train)/2)
    #train_length = int(len(subset_indices_train) / 4)
    #train_length = int(len(subset_indices_train) / 8)
    #train_length = int(len(subset_indices_train) / 16)



    train_loader = torch.utils.data.DataLoader(
        train_dataset_augmented, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train[:train_length])
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )
    
    '''
    # Load model
    model = PhotonNetS(batchNorm=False).to(device)



    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)


    '''
    # Training loop
    train_losses = []
    test_losses = []
    x = []
    fig, ax = plt.subplots(1)
    '''

    if False:
        for epoch in range(1, args.epochs + 1):
            #train and test each epoch
            train_loss = train(args, model, device, train_loader, optimizer, epoch)
            test_loss = test(model, device, val_loader)
            scheduler.step()    # learning rate scheduler

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            x.append(epoch - 1)
            ax.plot(x, test_losses, label='test_losses', markersize=2)
            ax.plot(x, train_losses, label='train_losses', markersize=2)

            plt.pause(0.05)

            # You may optionally save your model at each epoch here

        if args.save_model:

            print(train_losses)
            with open("train_losses_one.txt", "wb") as fp:  # Pickling
                pickle.dump(train_losses, fp)
            print(test_losses)
            with open("test_losses_one.txt", "wb") as fp:  # Pickling
                pickle.dump(test_losses, fp)



            torch.save(model.state_dict(), "mnist_model_onef.pt")


if __name__ == '__main__':
    main()

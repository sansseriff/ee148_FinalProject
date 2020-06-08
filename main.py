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
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
import pickle
from PIL import Image
import flow_transforms
from multiscaleloss import multiscaleEPE, realEPE, perImgEPE
from tensorboardX import SummaryWriter

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sn
import pandas as pd
from math import log
import math

from listdataset import ListDataset
from util2 import flow2rgb, AverageMeter, save_checkpoint
import os

random.seed(2020)
torch.manual_seed(2020)

evaluate = False
train = True
RGB = True

if RGB:
    dataset_mean = [0.6787816572469181, 25.33983741301654, 0.5781771226152195, 25.791804764729637, 0.473918180575716, 20.65891969206362]
    dataset_std = [0.40779226864547835, 31.37493267977034, 0.4592493703301828, 34.17082339570451, 0.42522918921017905, 30.06761831221986]
else:
    dataset_mean = [0.1906,4.622]
    dataset_std = [0.376,12.034]


# Training settings
# Use the command line to modify the default settings
parser = argparse.ArgumentParser(description='PyTorch PhotonNet')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size')
#parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=.1, metavar='LR',
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

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--multiscale-weights', '-w', default=[0.04, 0.08, 0.08, 0.16, 0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--milestones', default=[100, 150, 200], metavar='N', nargs='*',
                    help='epochs at which learning rate is divided by 2')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
#################################
parser.add_argument('--div-flow', default=.5,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
#################################
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')





best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def viewFlow(flow_array, dimx, dimy, to_8bit = False, filename = None, show = True):
    root = "..//data//output//testing_images"
    if to_8bit:
        flow_array = (flow_array * 255).astype(np.uint8)
    colormap = img2rgb(flow_array, dimx, dimy)
    im = Image.fromarray(colormap)
    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(im)
    if filename is not None:
        im.save(os.path.join(root,filename), quality=100)



def viewImg(img, to_8bit = False, Lum = False, filename = None, show = True):
    root = "..//data//output//testing_images"
    if to_8bit:
        im = Image.fromarray((img * 255).astype(np.uint8))
    else:
        im = Image.fromarray(img)

    if show:
        fig, ax = plt.subplots(1)
        if Lum:
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(im)
    if filename is not None:
        im.save(os.path.join(root,filename), quality=100)



def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()

    #rot = flow_transforms.RandomRotate(30)
    #trans = flow_transforms.RandomTranslate(5)

    for i, (input, target) in enumerate(train_loader):

        ######Data Augmentation#############
        #input, target = rot(input, target)
        #input, target = trans(input, target)


        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        #?????
        #input = torch.cat(input,1).to(device)
        input = input.to(device)


        #print("input is: ", input)
        #print("input[0] is: ", input[0])
        #print("input[0] size: ",input[0].size())

        if False:
            print("time input:")
            print(input[0,1,75:85,150:160])
            print("boolean input")
            print(input[0, 0, 75:85, 150:160])

        # compute output
        output = model(input)
        if args.sparse:
            # Since Target pooling is not very precise when sparse,
            # take the highest resolution prediction and upsample it instead of downsampling target
            h, w = target.size()[-2:]
            output = [F.interpolate(output[0], (h,w)), *output[1:]]

        loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)


        if False:
            print("type: ", type(output))
            print("length: ",len(output))
            print("it is: ",output[0])
            print("nextddddddddddddddddddddddd")
            print("it is: ", output[1])
            print("nextddddddddddddddddddddddd")
            print("it is: ", output[2])
            print("EPEEEEEEEEEEEEEEEEEEEEEEE")
            print(flow2_EPE)



        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses, flow2_EPEs))
        n_iter += 1
        if i >= epoch_size:
            break

        #if i > 5:
        #    break
    return losses.avg, flow2_EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    total_photons = []
    EPEs = []
    for i, (input, target) in enumerate(val_loader):


        if evaluate:
            #print(np.shape(input))
            #print(np.shape(input)[0])
            #print(input.size()[0])
            #print(np.shape(input.numpy()))
            #print("####################")
            photons = []
            for q in range(input.size()[0]):
                array = input.numpy()[q,0]
                array = array*dataset_std[0] + dataset_mean[0]
                sum = np.sum(array)
                photons.append(sum)

            total_photons.extend(photons)
        target = target.to(device)
        #################
        #?????????????????
        #input = torch.cat(input,1).to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        flow2_EPE = args.div_flow*realEPE(output, target, sparse=args.sparse)
        if evaluate:
            #perImgEPE(output, target, sparse=args.sparse).tolist()

            EPEs.extend(perImgEPE(output, target, sparse=args.sparse))

            #print("flow to epe: ", flow2_EPE.item())
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if False:
            if i < len(output_writers):  # log first output of first batches
                if epoch == 0:
                    mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
                    output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)

                    output_writers[i].add_image('Inputs', (input[0, :2].cpu() + mean_values).clamp(0, 1), 0)
                    output_writers[i].add_image('Inputs', (input[0, 2:].cpu() + mean_values).clamp(0, 1), 1)
                    #output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
                    #output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
                output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if epoch % 1 == 0 and i % 5 ==0:
            #try:
            a = flow2rgb(args.div_flow * output[0], max_value=10)
            #print(np.shape(a))
            a = np.transpose(a, (1, 2, 0))
            #print(np.shape(a))
            #print(a[15:19, 27:40])
            print(a.dtype)

            print("#######################")
            flow_map = output[0].detach().cpu().numpy()
            flow_map = np.transpose(flow_map, (1, 2, 0))
            print(np.shape(flow_map))

            IMG = ((a / np.max(a))*255).astype(np.uint8)


            viewImg(IMG, to_8bit = False, filename = "flow_" + str(i) + '.jpg', show = False)

            A = args.div_flow * flow_map

            #viewFlow(flow_array, dimx, dimy, to_8bit=False, filename=None, show=True):
            viewFlow(A, 64, 48, to_8bit = False, filename = 'gen_flow_' + str(i) + '.jpg', show = False)

            flow_map_gt = target[0].detach().cpu().numpy()
            flow_map_gt = np.transpose(flow_map_gt, (1, 2, 0))
            print(np.shape(flow_map_gt))
            viewFlow(flow_map_gt, 256, 192, to_8bit=False, filename = 'gt_flow_' + str(i) + '.jpg', show = False)

            #except:
             #   continue

            #viewImg(flow2rgb(args.div_flow * output[0], max_value=10))


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    if evaluate:
        print("length of total_photons: ", len(total_photons))
        print("length of EPEs: ", len(EPEs))
        fig = plt.figure()
        plt.plot(total_photons,EPEs,'.')
        plt.title("End Point Error vs Collected Photons")
        plt.xlabel("Photons in Example")
        plt.ylabel("EPE of Example")


    else:
        return flow2_EPEs.avg


def main():
    global args, best_EPE
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # this takes too long
    '''
    datasets_path = "C:\data\splits"
    test_data = np.load(os.path.join(datasets_path, "test_data.npy"))
    test_gt = np.load(os.path.join(datasets_path, "test_gt.npy"))
    train_data = np.load(os.path.join(datasets_path, "train_data.npy"))
    train_gt = np.load(os.path.join(datasets_path, "train_gt.npy"))

    print(np.shape(test_data))

    '''

    save_path = '..//data//output'
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))



    time_range = 64
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std),
        #transforms.Normalize(mean=[0.45,0.432,0.411], std=[1,1,1])
    ])
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    #co_transform = transforms.Compose([
    #    flow_transforms.RandomRotate(5),
    #    flow_transforms.RandomTranslate(10)
    #])

    #co_transform = flow_transforms.RandomRotate(10)

    if RGB:
        FM_1_testdir = "F://Flying_monkeys_2_RGB//test"
        FM_1_traindir = "F://Flying_monkeys_2_RGB//train"
    else:
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

    #print(train_list [787])
    #print(test_list[787])
    #train_dataset = ListDataset(FM_1_traindir , train_list, input_transform, target_transform, co_transform)
    #test_dataset = ListDataset(FM_1_testdir, test_list, input_transform, target_transform, co_transform)

    train_dataset = ListDataset(FM_1_traindir, train_list, input_transform, target_transform)
    test_dataset = ListDataset(FM_1_testdir, test_list, input_transform, target_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers = 2, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers = 2, pin_memory=True, shuffle=False)


    #a,b = train_dataset.__getitem__(178)

    #print(np.shape(a))
    #viewFlow(b,256,192)

    #np.shape(a)
    #print(a[: ,: ,0].dtype)
    #print(a[:, :, 1].dtype)
    #viewImg(a[:,:,1])
    #viewImg(a[:, :, 0])

    #print(a[100,100,0])
    #print(a[100, 100, 1])


    # Load model
    model = PhotonNetS(batchNorm=True).to(device)

    '''######################################################'''
    # Evaluate on the test set using saved model

    if evaluate:
        assert os.path.exists(os.path.join(save_path, "datset1_15epochs.pt"))

        # Set the test model
        model = PhotonNetS(batchNorm=True).to(device)

        model.load_state_dict(torch.load(os.path.join(save_path, "datset1_15epochs.pt")))



        # test_loader = torch.utils.data.DataLoader(
        #    test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        with torch.no_grad():
            epoch = 1
            EPE, total_photons = validate(val_loader, model, epoch, output_writers)
            print("EPE is:", EPE)
        return




    if train:

        # Try different optimzers here [Adam, SGD, RMSprop]
        #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.4)
        optimizer = optim.Adam(model.parameters(), 0.00100, betas=(.9, .999))

        # betas=(args.momentum, args.beta))

        # Set learning rate scheduler
        scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

        '''
        # Training loop
        train_losses = []
        test_losses = []
        x = []
        fig, ax = plt.subplots(1)
        '''
        train_EPE_list = []
        test_EPE_list = []
        epoch_list = []
        for epoch in range(21):
            epoch_list.append(epoch)

            # train for one epoch
            train_loss, train_EPE = train(train_loader, model, optimizer, epoch, train_writer)
            train_writer.add_scalar('mean EPE', train_EPE, epoch)
            train_EPE_list.append(train_EPE)

            # evaluate on validation set

            with torch.no_grad():
                EPE = validate(val_loader, model, epoch, output_writers)
                test_EPE_list.append(EPE)
            test_writer.add_scalar('mean EPE', EPE, epoch)

            scheduler.step()

            if best_EPE < 0:
                best_EPE = EPE

            is_best = EPE < best_EPE
            best_EPE = min(EPE, best_EPE)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "photonets",
                #'state_dict': model.module.state_dict(),
                'best_EPE': best_EPE,
                'div_flow': args.div_flow
            }, is_best, save_path)

        #if args.save_model:
            ######################
            # needs to be updated
            ######################
            #print(train_losses)
            #with open("train_losses_one.txt", "wb") as fp:  # Pickling
            #    pickle.dump(train_losses, fp)
            #print(test_losses)
            #with open("test_losses_one.txt", "wb") as fp:  # Pickling
            #    pickle.dump(test_losses, fp)
            print(train_EPE_list)
            print(test_EPE_list)
            train_EPE_ls = np.asarray(train_EPE_list)
            test_EPE_ls = np.asarray(test_EPE_list)
            np.save(os.path.join(save_path, "train_EPE_ls"), train_EPE_ls)
            np.save(os.path.join(save_path, "test_EPE_ls"), test_EPE_ls)
            fig = plt.figure()
            plt.plot(epoch_list, train_EPE_list, 'r', label="train")
            plt.plot(epoch_list, test_EPE_list, 'b', label="test")
            plt.title("End Point Error with Epoch")
            plt.legend()
            plt.show()

        torch.save(model.state_dict(), os.path.join(save_path, "datset1_10epochs_small_minibatch"))



if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F


def EPE(input_flow, target_flow, sparse=False, mean=True):

    if False:
        print("############################# input flow")
        print(input_flow.size())
        if input_flow.size()[2] == 192:
            print(input_flow[0,1,100:110,150:160])

        print("############################# target flow")
        print(target_flow.size())
        if target_flow.size()[2] == 192:
            print(target_flow[0,1,100:110,150:160])

    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def piEPE(input_flow, target_flow, sparse=False, mean=True):
    #EPEs for each image in the minibatch returned as a list

    if False:
        print("############################# input flow")
        print(input_flow.size())
        if input_flow.size()[2] == 192:
            print(input_flow[0,1,100:110,150:160])

        print("############################# target flow")
        print(target_flow.size())
        if target_flow.size()[2] == 192:
            print(target_flow[0,1,100:110,150:160])

    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        EPEs = []
        for img in range(EPE_map.size(0)):
            #if img == 0:
                #print(EPE_map[img].mean().detach().cpu().numpy())
            EPEs.append(EPE_map[img].mean().detach().cpu().numpy())
            #print(EPE_map[img].detach().cpu().numpy())
        return EPEs


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)

def perImgEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    return piEPE(upsampled_output, target, sparse, mean=False)

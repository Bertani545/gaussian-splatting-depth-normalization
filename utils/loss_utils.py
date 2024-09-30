#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def TVL(img, depths):

    channels, height, width = depths.size()
   
    assert channels == 1, "Input image must be gray scale"

    #min_val = depths.min()
    #max_val = depths.max()
    #depths = (depths - min_val) / (max_val - min_val)

    tv_h = (depths[:,1:,:] - depths[:,:-1,:]).abs().sum()
    tv_w = (depths[:,:,1:] - depths[:,:,:-1]).abs().sum()

    return (tv_h + tv_w)/(height * width)



def depth_distance(img):
    window_size = 3  # Adjust as needed
    kernel = torch.zeros(1, 1, window_size, window_size)

    # Center element: Multiplied by window_size * window_size
    kernel[0, 0, window_size // 2, window_size // 2] = window_size * window_size

    # Other elements: Set to -1
    kernel.fill_(-1)

    depth_distance = F.conv2d(img.unsqueeze(1), kernel, padding=window_size // 2)

    depth_distance = F.abs(depth_distance) / (window_size * window_size)

    # Perform element-wise multiplication and take absolute value
    result = torch.mean(depth_distance)

    return result.item()

def depth_distance_slow(img):

    window_size = 3  # Adjust as needed. Impar
    kernel = torch.zeros(1, 1, window_size, window_size)

    # Center element: Multiplied by window_size * window_size
    kernel[0, 0, window_size // 2, window_size // 2] = window_size * window_size

    # Other elements: Set to -1
    kernel.fill_(-1)


    result = 0

    radio = window_size//2

    for i in range(img.size(1)):
        for j in range(img.size(2)):
            
            temp_sum = 0

            for w_i in range(-radio, radio + 1):
                for w_j in range(-radio, radio + 1):

	                temp_sum += abs(img[0, i, j] - img[0, i + w_i, j + w_j])

            temp_sum = temp_sum / (window_size * window_size)

            result += temp_sum


    return result / (img.size(1) * img.size(2))



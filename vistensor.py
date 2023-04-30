import numpy as np
import cv2 as cv
import torch
import os
from networks import *


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = torch.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    cv.imshow('Tensor', grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":
    g_path = './models/AlexNet-2_gender_83.pt'

    g_net = None
    g_architecture = os.path.basename(g_path).split('_')[0].replace('-', '(') + ')'
    exec('g_net = ' + g_architecture)
    g_net.load_state_dict(torch.load(g_path, map_location=torch.device('cpu')))
    g_net.eval()

    layer = 1
    filter = g_net.features[layer].weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)
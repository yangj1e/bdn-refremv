from __future__ import print_function
import argparse
import os
import random
import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from math import log10
from PIL import Image

from dataset import ref_dataset
from vutil import save_image
import network

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument(
    '--batchSize', type=int, default=8, help='input batch size')
parser.add_argument(
    '--which_model_netG',
    type=str,
    default='cascade_unet',
    help='selects model to use for netG')
parser.add_argument(
    '--ns', type=str, default='5', help='number of blocks for each module')
parser.add_argument(
    '--netG', default='', help="path to netG (to continue training)")
parser.add_argument(
    '--norm',
    type=str,
    default='batch',
    help='instance normalization or batch normalization')
parser.add_argument(
    '--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument(
    '--imageSize',
    type=int,
    default=256,
    help='the height / width of the input image to network')
parser.add_argument(
    '--outf',
    default='.',
    help='folder to output images and model checkpoints')
parser.add_argument('--real', action='store_true', help='test real images')
parser.add_argument(
    '--iteration', type=int, default=0, help='number of iterative updates')
parser.add_argument(
    '--n_outputs', type=int, default=0, help='number of images to save')

opt = parser.parse_args()

str_ids = opt.ns.split(',')
opt.ns = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.ns.append(id)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

nc = 3
ngf = 64
netG = network.define_G(nc, nc, ngf, opt.which_model_netG, opt.ns, opt.norm,
                        opt.use_dropout, [], opt.iteration)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

transform = transforms.Compose([
    # transforms.Scale(opt.imageSize),
    # transforms.CenterCrop(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = ref_dataset(
    opt.dataroot,
    transform=transform,
    target_transform=transform,
    rf_transform=transform,
    real=opt.real)
assert dataset

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input = input.cuda()
netG.cuda()
netG.eval()

criterion = nn.MSELoss()
criterion.cuda()

for i, data in enumerate(dataloader, 1):
    if opt.real:
        input_cpu = data
        category = 'real'
    else:
        input_cpu, target_B_cpu, target_R_cpu = data
        category = 'test'
    input.resize_(input_cpu.size()).copy_(input_cpu)
    if opt.which_model_netG.startswith('cascade'):
        res = netG(input)
        if len(res) % 2 == 1:
            output_B, output_R = res[-1], res[-2]
        else:
            output_B, output_R = res[-2], res[-1]
    else:
        output_B = netG(input)

    if opt.n_outputs == 0 or i <= opt.n_outputs:
        save_image(output_B / 2 + 0.5, '%s/B_%04d.png' % (opt.outf, i))
        if opt.which_model_netG.startswith('cascade'):
            save_image(output_R / 2 + 0.5, '%s/R_%04d.png' % (opt.outf, i))

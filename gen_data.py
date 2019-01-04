from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
import random
from glob import glob
import argparse

import cv2
import scipy.misc
import numpy as np
from skimage import color

from PIL import Image

SIZES = (3, 5, 7)
SIGMAS = (0, 2)
THRESHOLDS = (0.2, 0.4)

def get_img_list(folders, ext='.jpg'):
    if ext is None:
        pattern = '*'
    else:
        pattern = '*' + ext
    return list(itertools.chain.from_iterable(glob(os.path.join(folder, pattern)) for folder in folders))


# img1 and img2 are PIL images
def sample_patches(img1, img2, size):
    w1, h1 = img1.size
    w2, h2 = img2.size
    if all(np.array((w1, h1, w2, h2)) >= 256):
        th = min(h1, h2)
        tw = min(w1, w2)
        x1 = random.randint(0, w1 - tw)
        y1 = random.randint(0, h1 - th)
        x2 = random.randint(0, w2 - tw)
        y2 = random.randint(0, h2 - th)
        img1 = img1.crop((x1, y1, x1 + tw, y1 + th))
        img2 = img2.crop((x2, y2, x2 + tw, y2 + th))
        return img1, img2
    else:
        return None


def sample_patch(img, crop_h, crop_w=None):
    if crop_w is None:
        crop_w = crop_h
    h, w, c = img.shape
    if h < crop_h or w < crop_w:
        return None
    j = random.randint(0, h - crop_h)
    i = random.randint(0, w - crop_w)
    return img[j:j + crop_h, i:i + crop_w, ...]


def merge(img1, img2, beta):
    return cv2.addWeighted(img1, 1 - beta, img2, beta, 0)


def generate_images(opt):
    if not opt.test:
        train_list_f = os.path.join(opt.dataroot, 'ImageSets', 'Main', 'train.txt')
    else:
        train_list_f = os.path.join(opt.dataroot, 'ImageSets', 'Main', 'val.txt')
    with open(train_list_f) as f:
        train_list = f.read().splitlines()

    obs_dir = os.path.join(opt.outf, 'obs')
    trans_dir = os.path.join(opt.outf, 'trans')
    ref_dir = os.path.join(opt.outf, 'ref')
    refb_dir = os.path.join(opt.outf, 'refb')
    # label_dir = os.path.join(opt.outf, 'label')

    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)
    if not os.path.exists(obs_dir):
        os.mkdir(obs_dir)
    if not os.path.exists(trans_dir):
        os.mkdir(trans_dir)
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(refb_dir):
        os.mkdir(refb_dir)
    # if not os.path.exists(label_dir):
    #     os.mkdir(label_dir)
    print('Number of source images: %d' % len(train_list))

    # random_crop = transforms.RandomCrop(opt.imageSize)
    # f = open(os.path.join(opt.outf, 'stat.txt'), 'w')
    for i in range(opt.numImages):
        while True:
            T_f, R_f = random.choices(train_list, k=2)
            T = np.array(Image.open(os.path.join(opt.dataroot, 'JPEGImages', T_f + '.jpg')))
            R = np.array(Image.open(os.path.join(opt.dataroot, 'JPEGImages', R_f + '.jpg')))
            T_crop = sample_patch(T, opt.imageSize)
            R_crop = sample_patch(R, opt.imageSize)
            if T_crop is not None and R_crop is not None:
                break
            # patches = sample_patches(T, R, opt.imageSize)
            # if patches is not None:
            #     T_crop, R_crop = patches
            #     break
        # T_crop = np.array(T_crop)
        # R_crop = np.array(R_crop)
        beta = random.uniform(*THRESHOLDS)
        sigma = random.uniform(*SIGMAS)
        size = random.choice(SIZES)
        R_blur = cv2.GaussianBlur(R_crop, (size, size), sigma)
        I = merge(T_crop, R_blur, beta)
        scipy.misc.imsave(os.path.join(obs_dir, '{:06d}.jpg'.format(i + 1)), I)
        scipy.misc.imsave(os.path.join(trans_dir, '{:06d}.jpg'.format(i + 1)), T_crop)
        scipy.misc.imsave(os.path.join(ref_dir, '{:06d}.jpg'.format(i + 1)), R_crop)
        scipy.misc.imsave(os.path.join(refb_dir, '{:06d}.jpg'.format(i + 1)), R_blur)
        # f.write('{}\t{}\t{}\t{}\t{}\n'.format(T_f, R_f, beta, size, sigma))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to BSDS500 dataset')
    parser.add_argument('--outf', required=True, help='folder to output generated dataset')
    parser.add_argument('--numImages', type=int, default=10000, help='number of images to generate')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the image')
    parser.add_argument('--test', action='store_true', help='generate test images')
    opt = parser.parse_args()
    print(opt)

    generate_images(opt)

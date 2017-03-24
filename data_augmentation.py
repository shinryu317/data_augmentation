# -*- coding: utf-8 -*-
'''
data_augmentation.py
==============================
Data augmentation with affine transformation, gamma correction and mirror operation.
'''
from __future__ import print_function
import argparse
import pickle
import json
import os

import cv2
import numpy as np

from transformer import DataTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file .json.')
    return parser.parse_args()

def loadtxt(config):
    path_list = []
    with open(config['data_file'], 'r') as f:
        for line in f:
            i, j = line.replace(',', ' ').split()
            image = os.path.join(config['image_path'], i)
            label = os.path.join(config['label_path'], j) if not j.isdigit() else int(j)
            path_list.append((image, label))
    return path_list

def building(transformer, data, savefile):
    images, labels = [], []
    for image_file, label in data:
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if isinstance(label, str):
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        image, label = transformer.transform(image, label)
        images.append(image)
        labels.append(label)
    save(savefile[0], np.asarray(images, dtype=np.uint8))
    save(savefile[1], np.asarray(labels, dtype=np.int32))

def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def main():
    args = parse_args()
    config = json.load(open(args.config_file, 'r'))

    path_list = loadtxt(config['input'])
    assert 0 < path_list

    save_dir = config['output']['save_dir']
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print('Transformation parameters.')
    params = config['parameters']
    for key in params.keys():
        print('  => {}: {}'.format(key, params[key]))

    print('Data augmentation.')
    transformer = DataTransformer(config)
    x_times_sample = params['x_times']
    for i in range(x_times_sample):
        data = np.random.permutation(path_list)
        savefile = [os.path.join(save_dir, '{:04d}_x.pkl'.format(i)),
                    os.path.join(save_dir, '{:04d}_y.pkl'.format(i))]
        building(transformer, data, savefile)
        print('  => # of completed x times sample: {} / {}'.format(i + 1, x_times_sample), end='\r')
    print('\nDone')

if __name__ == '__main__':
    main()
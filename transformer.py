# -*- coding: utf-8 -*-
import cv2
import numpy as np

from param_generator import ParamGenerator


class DataTransformer(object):

    def __init__(self, config):
        out_config = config['output']
        self.out_shape = tuple(out_config['shape'])
        self.color_type = out_config['color_type']
        self.out_keys = out_config.keys()
        self.param_generator = ParamGenerator(config['parameters'])

    def transform(self, image, label):
        params = self.param_generator.generate()
        if isinstance(label, np.ndarray):  # segmentation task
            image = _image_transform(image, params, interpolation=cv2.BORDER_REPLICATE)
            label = _image_transform(label, params, interpolation=cv2.BORDER_REPLICATE)
            if 'shape' in self.out_keys:
                label = cv2.resize(label, self.out_shape, interpolation=cv2.INTER_NEAREST)
        else:  # classification task
            image = _image_transform(image, params, cv2.BORDER_TRANSPARENT)
            if 'shape' in self.out_keys:
                image = cv2.resize(image, self.out_shape)
            if self.color_type == 'gray':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image, label

def _image_transform(image, params, interpolation):
    scale, shift_radius, angle, gamma_table, flip = params
    canvas = cv2.LUT(image, gamma_table).astype(np.uint8)
    if flip:
        canvas = cv2.flip(canvas, 1)
    axis = canvas.shape[0] / 2, canvas.shape[1] / 2
    rot_mat = cv2.getRotationMatrix2D(axis, angle, scale)
    rot_mat[:, 2] += shift_radius['x'], shift_radius['y']
    return cv2.warpAffine(canvas, rot_mat, canvas.shape[:2], borderMode=interpolation)
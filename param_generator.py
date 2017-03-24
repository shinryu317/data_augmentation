# -*- coding: utf-8 -*-
import numpy as np


class ParamGenerator(object):

    def __init__(self, params):
        self.scale  = params['scale']  if 'scale'  in params.keys() else (1.0, 1.0)
        self.radius = params['radius'] if 'radius' in params.keys() else (0.0, 0.0)
        self.angle  = params['angle']  if 'angle'  in params.keys() else (0.0, 0.0)
        self.gamma  = params['gamma']  if 'gamma'  in params.keys() else (1.0, 1.0)
        self.flip   = params['flip']   if 'flip'   in params.keys() else False

    def generate(self):
        return [_generate_scale_rate(self.scale),
                _generate_shift_radius(self.radius),
                _generate_rotation_angle(self.angle),
                _generate_gamma_table(self.gamma),
                _generate_flip_flag(self.flip)]

def _generate_scale_rate(scale):
    return np.random.uniform(min(scale), max(scale))

def _generate_shift_radius(radius):
    # numpy's randint is return [min, max).
    min_radius, max_radius = min(radius), max(radius) + 1
    shift_x = np.random.randint(min_radius, max_radius)
    shift_y = np.random.randint(min_radius, max_radius)
    return {'x': shift_x, 'y': shift_y}

def _generate_rotation_angle(angle):
    return np.random.randint(min(angle), max(angle))

def _generate_gamma_table(gamma):
    gamma = np.random.uniform(min(gamma), max(gamma))
    table = np.arange(256).astype('float')
    table = 255 * (table / 255) ** (1.0 / gamma)
    return table

def _generate_flip_flag(flip):
    return np.random.choice((True, False)) if flip else False
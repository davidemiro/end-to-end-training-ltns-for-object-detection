"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import configparser
import numpy as np
import keras
from ..utils.anchors import AnchorParameters


# read config file in .ini format
# manages only anchor parameters
def read_config_file(config_path):
    config = configparser.ConfigParser()

    with open(config_path, 'r') as file:
        config.read_file(file)

    assert 'anchor_parameters' in config, \
        "Malformed config file. Verify that it contains the anchor_parameters section."

    config_keys = set(config['anchor_parameters'])
    default_keys = set(AnchorParameters.default.__dict__.keys())

    assert config_keys <= default_keys, \
        "Malformed config file. These keys are not valid: {}".format(config_keys - default_keys)

    return config


def parse_anchor_parameters(config):
    ratios = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)


class Config:

    def __init__(self):
        self.lab_experiment_path = './labnotes'
        self.snapshot = None
        self.imagenet_weights = True
        self.weights = ''
        self.backbone = 'resnet50'
        self.gpu = 0
        # Training params
        self.batch_size = 1
        self.initial_epoch = 0  # for iterative training
        self.epochs = 80
        self.steps = 500
        self.lr = 1e-5

        # Anchor Parameters
        self.anchor_box_sizes = [32, 64, 128, 256, 512]
        self.anchor_box_strides = [8, 16, 32, 64, 128]
        self.anchor_box_ratios = np.array([0.5, 1, 2], keras.backend.floatx())
        self.anchor_box_scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())

        self.nms_threshold = 0.7

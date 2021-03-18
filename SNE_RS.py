import os
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2labelim, tensor2confidencemap
from models.sne_model import SNE
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2 as cv

class Options():
    def initialize(self, parser):
        self.phase = 'test'
        parser.add_argument('--prob_map', action='store_true', help='chooses outputting prob maps or binary predictions')
        parser.add_argument('--no_label', action='store_true', help='chooses if we have gt labels in testing phase')
        self.isTrain = False
        return parser
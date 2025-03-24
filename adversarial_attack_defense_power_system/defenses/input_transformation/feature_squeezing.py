import numpy as np
import tensorflow as tf
from art.defences.preprocessor import FeatureSqueezing


def feature_squeezing(train_data, bit_depth=3):
    fs = FeatureSqueezing(bit_depth=bit_depth, clip_values=(-5, 5))
    fs_x = fs(train_data)[0]
    return fs_x

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.platform import gfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# boilerplate code
import os
from io import BytesIO
import numpy as np
import random
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
import somoclu
from sklearn.metrics.pairwise import pairwise_distances
# import deep_som as ds
import time
import matplotlib.cm as cm
from pathlib import Path
import pprint, pickle
import neural_map as nm
import utils

from Speech_command_classification.convRNN import MultiConvRNN



rnn = MultiConvRNN()


np.set_printoptions(precision=3)
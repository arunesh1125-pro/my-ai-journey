import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("STARTING SCRIPT...")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras import layers

print("TensorFlow version:", tf.__version__)
print("TensorFlow works perfectly ✅")
print("Keras layer:", layers.Dense)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Available Devices:")
print(tf.config.list_physical_devices())
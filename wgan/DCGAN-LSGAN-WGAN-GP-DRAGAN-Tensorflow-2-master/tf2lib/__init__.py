import tensorflow as tf

from tf2lib.data import *
from tf2lib.image import *
from tf2lib.ops import *
from tf2lib.utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


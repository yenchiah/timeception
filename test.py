import numpy as np
import torch as T
from nets import timeception_pytorch


# i3d first input layer has shape (10, 3, 36, 224, 224)
# (batch_size, channel, time, height, width)

# define input tensor (from i3d last layer before logic)
input = T.tensor(np.zeros((10, 1024, 5, 7, 7)), dtype=T.float32)

# define 4 layers of timeception
module = timeception_pytorch.Timeception(input.size(), n_layers=2)

# feedforward the input to the timeception layers
tensor = module(input)

# the output is (10, 1600, 1, 7, 7)
print (tensor.size())

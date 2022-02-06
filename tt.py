import torch
import numpy as np
# import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as functional
from torchvision import transforms, datasets

a = np.array([[0,1], [1,2]])

scalar1 = torch.tensor([1.])
print(scalar1)

scalar2 = torch.tensor([3.])
print(scalar2)


add_scalar = scalar1+scalar2
print(add_scalar)
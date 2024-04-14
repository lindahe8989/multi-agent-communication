import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
from mambabyte import MambaConfig, Mamba
from torchvision import datasets, transforms, ops
import matplotlib.pyplot as plt

# from mamba_ssm import Mamba

transform = transforms.Compose([transforms.ToTensor(), ops.Permute([1, 2, 0])])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=4, shuffle=True)

# Display a few images
data_iter = iter(data_loader)
images, labels = next(data_iter)
images = torch.tile(images, dims=(3,))
plt.imshow(images[0])

flattened_images = images[0].view(-1, 3)  # Reshape to (784, 3)
int_pixels = (flattened_images * 256).int() 
int_pixels = torch.clamp(int_pixels, 0, 255)
one_hot = F.one_hot(int_pixels, 256)

config = MambaConfig(
    dim = 256,
    depth = 3,
    dt_rank = 16,
    d_state = 2,
    expand_factor = 2,
    d_conv = 3,
    dt_min = 0.001,
    dt_max = 0.1,
    dt_init = "random",
    dt_scale = 1.0,
    bias = False,
    conv_bias = True,
    pscan = True
)

model = Mamba(config)
out = model(one_hot)
print(out)
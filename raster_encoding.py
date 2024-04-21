"""
pip install mamba-ssm causal-conv1d>=1.2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mamba_ssm import Mamba
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

assert torch.cuda.is_available()
device = torch.device("cuda")

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

class ToRGB(nn.Module):
  def forward(self, img):
    return to_tensor(img).permute([1, 2, 0]).tile(dims=(3,))

def expand_bits(x):
  return 1 & (x.unsqueeze(-1) >> torch.arange(7, -1, -1))

class ToBits(nn.Module):
  def forward(self, x):
    x = torch.clamp((x * 256).int(), 0, 255)
    x = expand_bits(x).flatten()
    return F.one_hot(x, 2)

batch_size = 4
transform = transforms.Compose([ToRGB(), ToBits()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class Raster(nn.Module):
  def __init__(self, input_dim=2, hidden_dim=16, output_dim=2, d_state=16, d_conv=4, expand=2, mamba_blocks=4):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.d_conv = d_conv
    self.expand = expand
    self.mamba_blocks = mamba_blocks
    if input_dim != hidden_dim:
      self.fc_in = nn.Linear(input_dim, hidden_dim)
      self.fc_out = nn.Linear(hidden_dim, output_dim)
    else:
      self.fc_in = nn.Identity()
      self.fc_out = nn.Identity()
    self.layers = nn.ModuleList([
        Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        for i in range(mamba_blocks)
    ])

  def forward(self, x, inference_params=None):
    x = self.fc_in(x)
    if inference_params is None:
      inference_params = [None for i in self.layers]
    for (layer, inf_param) in zip(self.layers, inference_params):
      x = x + layer(x, inf_param)
    x = self.fc_out(x)
    return x

def train_raster(raster, optim, epochs=10):
  raster.train()

  for epoch in range(epochs):
    total_loss = 0
    pbar = tqdm(train_loader)
    for batch_idx, (target, labels) in enumerate(pbar):
      target = target.float().to(device)
      optim.zero_grad()

      next_prediction = raster(target)
      loss = F.cross_entropy(next_prediction, target)
      
      loss.backward()
      optim.step()

      total_loss += loss.item()
      if batch_idx % 1 == 0:
        pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(target), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), total_loss / (1 + batch_idx)))

raster = Raster(mamba_blocks=3).to(device)
optim = torch.optim.Adam(raster.parameters(), lr=1e-3)

# Stop early, it'd take hours to actually train this and you get good results in just 30 seconds.
train_raster(raster, optim, 10)

with torch.no_grad():
  x = next(iter(train_loader))[0][0].float()
  print(x[..., 0].mean())
  y = raster(x.unsqueeze(0).float().to(device))[0]
  img = y.argmax(dim=-1).double().reshape(28, 28, 3, 8) @ (1 << torch.arange(7, -1, -1)).double().to(device)
  img = img.int().cpu()
  plt.imshow(img)
  plt.show()
  F.softmax(y, dim=-1)
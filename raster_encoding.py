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

class ToBytes(nn.Module):
  def forward(self, x):
    x = torch.clamp((x * 256).long(), 0, 255)
    x = x.flatten()
    return F.one_hot(x, 256)

batch_size = 4
transform = transforms.Compose([ToRGB(), ToBits()])
# transform = transforms.Compose([ToRGB(), ToBytes()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class Raster(nn.Module):
  def __init__(self, input_dim=2, hidden_dim=16, output_dim=2, d_state=16, d_conv=4, expand=2, mamba_blocks=4):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.d_state = d_state
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

  def forward(self, x):
    x = self.fc_in(x)
    for layer in self.layers:
      x = x + layer(x)
    x = self.fc_out(x)
    return x

  def step(self, x, conv_state, ssm_state):
    x = self.fc_in(x)
    new_conv_state = []
    new_ssm_state = []
    for (layer, conv, ssm) in zip(self.layers, conv_state, ssm_state):
      dx, new_conv, new_ssm = layer.step(x, conv, ssm)
      new_conv_state.append(conv)
      new_ssm_state.append(ssm)
      x = x + dx
    x = self.fc_out(x)
    return x, new_conv_state, new_ssm_state

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
train_raster(raster, optim, 10)

with torch.no_grad():
  x = next(iter(train_loader))[0][0].float()
  print(x[..., 0].mean())
  y = raster(x.unsqueeze(0).float().to(device))[0]
  img = y.argmax(dim=-1).double().reshape(28, 28, 3, 8) @ (1 << torch.arange(7, -1, -1)).double().to(device)
  img = F.softmax(y, dim=-1)[..., 1].double().reshape(28, 28, 3, 8) @ (1 << torch.arange(7, -1, -1)).double().to(device)
  img = img.int().cpu()
  plt.imshow(img)
  plt.show()

def sample(raster, beta=0.3):
  x = torch.tensor([[[1., 0.]]]).to(device)
  conv_state = torch.zeros(
      raster.mamba_blocks, 1, raster.hidden_dim * raster.expand, raster.d_conv, device=device
  )
  ssm_state = torch.zeros(
      raster.mamba_blocks, 1, raster.hidden_dim * raster.expand, raster.d_state, device=device
  )

  bits = [0]
  with torch.no_grad():
    pbar = tqdm(range(28 * 28 * 3 * 8 - 1))
    for i in pbar:
      y, conv_state, ssm_state = raster.step(x, conv_state, ssm_state)
      p = torch.softmax(beta * y[0][0], -1)
      # print(y[0][0], p)
      idx = torch.multinomial(p, 1)
      bits.append(idx.item())
      x = F.one_hot(idx, 2).unsqueeze(0).float()
  return bits

# To be continued...
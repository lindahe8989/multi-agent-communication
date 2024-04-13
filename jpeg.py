import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from io import BytesIO
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

def lossy(img, quality=75):
  """
  Inputs:
    - img - An image tensor.
    - quality - Compression quality. Higher = more bits.
  
  Returns:
    - Lossy reconstruction (as torch tensor),
    - Length of encoding (in bytes).

  Credits to https://stackoverflow.com/a/77893423/8387437
  """
  buffer = BytesIO()
  pil_img = to_pil(img.permute(2, 0, 1))
  pil_img.save(buffer, "JPEG", quality=quality, optimize=True)
  return to_tensor(Image.open(buffer)).permute(1, 2, 0), buffer.tell()

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

class ToRGB(nn.Module):
  def forward(self, img):
    return to_tensor(img).permute([1, 2, 0]).tile(dims=(3,))

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=ToRGB())

bit_losses = dict((i, []) for i in range(1000))
for quality in range(0, 100, 1):
  for i, (img, label) in enumerate(mnist_data):
    reconstruction, length = lossy(img, quality)
    loss = F.mse_loss(img, reconstruction).item()
    
    bit_losses[length].append(loss)
    print(quality, i, length, loss)
    if i > 100:
      break

bits = []
losses = []
for x, y in bit_losses.items():
  y = np.mean(y)
  if np.isfinite(y):
    bits.append(x)
    losses.append(y)

plt.scatter(bits, losses)
plt.yscale('log')

plt.title("JPEG Compression")
plt.xlabel("Bits")
plt.ylabel("Log MSE Error")
plt.show()
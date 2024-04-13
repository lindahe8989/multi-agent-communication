import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np

from torchvision import datasets, transforms, ops
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), ops.Permute([1, 2, 0])])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=4, shuffle=True)

# Display a few images
data_iter = iter(data_loader)
images, labels = next(data_iter)
images = torch.tile(images, dims=(3,))
plt.imshow(images[0])
# print(torch.amax(images[0]))
print(images[0])
print(images[0].shape)
# torch.Size([28, 28, 3])
## discretize so that for each pixel, the 3-vector is transformed into a 3 * 8 vector, where each vector of dimension 8 and each is either 0 or 1 
flattened_images = images[0].view(-1, 3)  # Reshape to (784, 3)
int_pixels = (flattened_images * 256).int() 
int_pixels = torch.clamp(int_pixels, 0, 255)
print(torch.amax(int_pixels))

print(int_pixels)
print(int_pixels.shape)
torch.Size([784, 3])
int_pixels_compressed = int_pixels.view(-1, 1)

class PixelTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers):
        super(PixelTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model = model_dim, nhead = nhead, num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers)
        self.decoder = nn.Linear(model_dim, input_dim)

    def forward(self, src): 
        src = src.float()
        src = self.embedding(src)  
        output = self.transformer(src, src)
        output = self.decoder(output)
        return output 

pixel_transformer = PixelTransformer(input_dim=1, model_dim=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
# # Add a batch dimension (with size 1) to match the expected input shape [sequence length, batch size, features]
output = pixel_transformer(int_pixels_compressed)
print(output)
print(output.shape)

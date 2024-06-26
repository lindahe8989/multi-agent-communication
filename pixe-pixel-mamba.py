import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer 
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
# print(torch.amax(images[0]))
print(images[0])
print(images[0].shape)
# torch.Size([28, 28, 3])
## discretize so that for each pixel, the 3-vector is transformed into a 3 * 8 vector, where each vector of dimension 8 and each is either 0 or 1 
flattened_images = images[0].view(-1, 3)  # Reshape to (784, 3)
int_pixels = (flattened_images * 256).int() 
int_pixels = torch.clamp(int_pixels, 0, 255)
print(torch.amax(int_pixels))

def int_to_base2_vector(int_tensor):
    base2_tensor = torch.stack([(int_tensor >> i) & 1 for i in range(7, -1, -1)], dim=-1)
    return base2_tensor

int_pixels_base2 = int_to_base2_vector(int_pixels)
print(int_pixels_base2.shape)
print(int_pixels_base2)
int_pixels_concatenated = int_pixels_base2.reshape(784, -1)
int_pixels_flattened = int_pixels_concatenated.view(-1, 1)
print(int_pixels_flattened.shape)
int_pixels_flattened = int_pixels_flattened.long()
print(int_pixels_flattened.dtype)
# torch.int32

# class PixelTransformer(nn.Module):
#     def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers):
#         super(PixelTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, model_dim)
#         self.transformer = nn.Transformer(d_model = model_dim, nhead = nhead, num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers)
#         self.decoder = nn.Linear(model_dim, input_dim)

#     def forward(self, src): 
#         src = src.float()
#         src = self.embedding(src)  
#         output = self.transformer(src, src)
#         output = self.decoder(output)
#         return output 


# I'd use MambaByte: https://huggingface.co/papers/2401.13660
# You'll have to change from bits to bytes = 8 bits and use one-hot encodings.
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
# # input_ids = tokenizer("Hey how are you doing?", return_tensors = "pt")["input_ids"]
# # input_ids (torch.LongTensor of shape (batch_size, input_ids_length)) — Indices of input sequence tokens in the vocabulary.
# # Example parameters
# vocab_size = 30000  # Adjust based on the actual vocabulary size
# batch_size = 1
# input_ids_length = 10

# # Generate random input_ids
# random_input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, input_ids_length), dtype=torch.long)

# out = model.generate(random_input_ids, max_new_tokens = 10)
out = model.generate(int_pixels_flattened, max_new_tokens = 10)
print(out)



# pixel_transformer = PixelTransformer(input_dim=1, model_dim=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
# # # Add a batch dimension (with size 1) to match the expected input shape [sequence length, batch size, features]
# output = pixel_transformer(int_pixels_compressed)
# print(output)
# print(output.shape)

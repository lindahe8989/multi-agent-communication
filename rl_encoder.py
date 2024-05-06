# !pip install mamba-ssm causal-conv1d>=1.2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mamba_ssm import Mamba
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm.notebook import tqdm
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
from torch.distributions import Categorical
from torchvision import models, transforms

assert torch.cuda.is_available()
device = torch.device("cuda")

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

batch_size = 1
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=to_tensor
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


class DeepMamba(nn.Module):
    def __init__(self, d_size=16, d_state=16, d_conv=4, expand=2, n_layers=5):
        super().__init__()
        self.d_size = d_size
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for i in range(n_layers)
            ]
        )

    def step(self, x):
        new_conv_state = []
        new_ssm_state = []
        for layer, conv, ssm in zip(self.mamba_layers, self.conv_state, self.ssm_state):
            dx, new_conv, new_ssm = layer.step(x, conv, ssm)
            new_conv_state.append(conv)
            new_ssm_state.append(ssm)
            x = x + dx
        self.conv_state = new_conv_state
        self.ssm_state = new_ssm_state
        return x

    def reset(self):
        """
        Prepares it for a new round of inputs.
        """
        self.conv_state = torch.zeros(
            self.n_layers, 1, self.hidden_dim * self.expand, self.d_conv, device=device
        )
        self.ssm_state = torch.zeros(
            self.n_layers, 1, self.hidden_dim * self.expand, self.d_state, device=device
        )
        return self.ssm_state[-1]


class Upscaler(nn.Module):
    """
    hidden_size latent space ->
    32 x 4 x 4 representation ->
    16 x 10 x 10 representation ->
    1 x 28 x 28 (greyscale) image.
    """

    def __init__(self, hidden_size=512, output_size=28):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 32 * ((output_size + 8) // 9) ** 2)
        self.deconvolutions = nn.ModuleList(
            [
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3, padding=1),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=3, padding=1),
            ]
        )

    def forward(self, x):
        x = F.elu(self.fc(x.flatten(start_dim=1)))
        x = x.view(x.shape[0], 32, 4, 4)
        for deconv in self.deconvolutions:
            x = F.elu(deconv(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, img_size=28, n_layers=5):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.mamba = DeepMamba(d_size=hidden_dim, n_layers=n_layers)
        hidden_size = hidden_dim * self.mamba.d_state * self.mamba.expand
        self.upscaler = Upscaler(hidden_size=hidden_size)

    def step(self, x):
        x = self.fc_in(x)
        self.mamba.step(x)
        ssm = self.mamba.ssm_state[-1]
        return self.upscaler(ssm)

    def reset(self):
        """
        Prepares it for a new (encoding of an) image.
        """
        ssm = self.mamba.reset()
        return self.upscaler(ssm)


class Downscaler(nn.Module):
    """
    1 x 28 x 28 image ->
    16 x 10 x 10 representation ->
    32 x 4 x 4 representation ->
    hidden_size latent space.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=1),
            ]
        )
        conv_size = 32 * ((input_size + 8) // 9) ** 2
        self.fc = nn.Linear(conv_size, hidden_size)

    def forward(self, x):
        for conv in self.convolutions:
            x = F.elu(conv(x))
        x = x.view(x.shape[0], -1)
        return F.elu(self.fc(x))


class QNetwork(nn.Module):
    def __init__(self, input_size=28, hidden_dim=16, num_actions=3, n_layers=3):
        super().__init__()
        self.features = Downscaler(input_size=input_size, hidden_size=hidden_dim)
        self.mamba = DeepMamba(d_size=hidden_dim, n_layers=n_layers)
        self.fc = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)]
        )
        self.fc_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.features(x)
        x = self.mamba(x)
        for fc in self.fc:
            x = x + F.elu(fc(x))
        return F.elu(self.fc_out(x))


def train_policy(
    qnetwork,
    optim,
    decoder,
    epochs=1,
    max_images=100,
    max_bits=5,
    gamma=0.99,
    beta=1,
    bit_penalty=1e-2,
):
    """
    qnetwork - The RL qnetwork.
    optim - The optimizer for policy.
    decoder - A frozen decoder model.
    epochs - Number of run-throughs of the training data.
    max_bits - Largest encoding output.
    beta - Inverse temperature (controls)
    bit_penalty - Penalty for each additional bit.
    """
    qnetwork.train()
    decoder.eval()

    total_images = 0
    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        for batch_idx, (img, label) in enumerate(pbar):
            total_images += 1
            if total_images == max_images:
                break
            img = img.to(device)
            optim.zero_grad()
            with torch.no_grad():
                reconstruction = decoder.reset()
                error = img - reconstruction

            i = 0
            going = True
            while i < max_bits and going:
                i += 1
                qvalues = qnetwork(error)
                probs = F.softmax(beta * qvalues, dim=-1)
                action = torch.multinomial(probs, 1)
                idx = action.item()
                if idx == 2:
                    qvalue = -F.mse_loss(error, 0 * error)
                    going = False
                else:
                    one_hot = F.one_hot(action, 2).float()
                    with torch.no_grad():
                        reconstruction = decoder.step(one_hot)
                        error = img - reconstruction
                    reward = -F.mse_loss(error, 0 * error) - bit_penalty
                    with torch.no_grad():
                        futures = qnetwork(error)
                    qvalue = reward + gamma * futures.max()
                loss = (qvalues[..., idx] - qvalue) ** 2
                loss.backward()
                optim.step()

                pbar.set_description(
                    "Image #{}, Bit {}, Error {:.6f}, QValues {},  Qvalue {:.3f}".format(
                        batch_idx,
                        i,
                        F.mse_loss(error, 0 * error).item(),
                        qvalues.squeeze().detach().cpu().numpy(),
                        qvalue.item(),
                    )
                )


def train_decoder(
    decoder, optim, qnetwork, epochs=1, max_images=100, max_bits=5, beta=1
):
    qnetwork.eval()
    decoder.train()

    total_images = 0
    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        for batch_idx, (img, label) in enumerate(pbar):
            total_images += 1
            if total_images == max_images:
                break
            img = img.to(device)
            optim.zero_grad()
            reconstruction = decoder.reset()
            error = img - reconstruction
            loss = F.mse_loss(error, 0 * error)
            loss.backward()
            optim.step()
            optim.zero_grad()

            for i in range(max_bits):
                with torch.no_grad():
                    qvalues = qnetwork(error)
                    probs = F.softmax(beta * qvalues, dim=-1)
                    action = torch.multinomial(probs, 1)
                    idx = action.item()
                if idx == 2:
                    break

                one_hot = F.one_hot(action, 2).float()
                reconstruction = decoder.step(one_hot)
                error = img - reconstruction
                loss = F.mse_loss(error, 0 * error)
                loss.backward()
                optim.step()

                pbar.set_description(
                    "Image #{}, Bit {}, Error {:.6f}, QValues {}".format(
                        batch_idx,
                        i,
                        F.mse_loss(error, 0 * error).item(),
                        qvalues.squeeze().detach().cpu().numpy(),
                    )
                )


qnetwork = QNetwork().to(device)
q_optim = torch.optim.Adam(qnetwork.parameters(), lr=1e-3)

decoder = Decoder().to(device)
d_optim = torch.optim.Adam(decoder.parameters(), lr=1e-3)


"""
Slowing increasing beta during training should make it converge.
Also, we'll need to slowly add more bits/images.
"""

for i in range(5):
    train_policy(qnetwork, q_optim, decoder)
    train_decoder(decoder, d_optim, qnetwork)

# Preprocessing 
# - Convert images into a suitable format for the RL agent, possibly resizing or normalizing them 

import torch 
import torchvision.transforms as transforms 
from torchvision.models import resnet18 
from PIL import Image 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import os 
import torchvision.models as models 

def preprocess_image(image_path): 
    """
    Load an image, resize and normalize it. 
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize image to 224x224 for ResNet 
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image 

# Feature Extraction
# - Use a pre-trained CNN (Convolutional Neural Network) to extract features from the images. 
# - These features can serve as the state representation for the RL agent 

def extract_features(image_tensor): 
    """
    Extract features from an image using a pre-trained CNN. 
    """
    model = resnet18(pretrained=True)
    model.eval() # Set model to evaluation mode 
    with torch.no_grad(): 
        features = model(image_tensor)
    return features 


# RL Agent Design 
# - Design an RL agent that takes the state as input and decides on the next action to take 
# - The action could be selecting a set of parameters for a specific compression algorithm 


# Policy Network:
# - A simple neural network that takes the state (image features) as input and outputs a probability distribution over actions 
# - The action space size (output_size) depends on how you define the compression parameters or techniques 

class PolicyNetwork(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size), 
            nn.Softmax(dim=-1)
        )

    def forward(self, x): 
        return self.network(x)


# Different Action Spaces 
# class CompressionActionSpace: 
#     def __init__(self): 
#         self.actions = {
#             0: {'quantization_level': 'low', 'encoding_strategy': 'Huffman'}, 
#             1: {'quantization_level': 'medium', 'encoding_strategy': 'Arithmetic'}, 
#             2: {'quantization_level': 'high', 'encoding_strategy': 'RunLength'}, 
#         }

#     def get_action_params(self, action_id): 
#         return self.actions[action_id]
    

class RLAgent: 
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 1e-3): 
        self.policy_network = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.saved_log_probs = []
        self.rewards = []

    # Samples an action according to the probability distribution output by the policy network
    # It saves the log probability of the selected action for training 
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        action = np.random.choice(len(probs.detach().numpy()[0]), p = probs.detach().numpy()[0])
        self.saved_log_probs.append(torch.log(probs.squeeze(0)[action]))
        return action 
    
    # Implements the policy gradient update rule. 
    # It calculates the discounted rewards and scales them to have a mean of 0 and a standard deviation of 1. 
    # Then it computes the policy gradient loss and performs a backpropagation step. 
    def update_policy(self):
        R = 0 
        policy_loss = []
        rewards = []
        # Discount future rewards back to the present using gamma 
        for r in self.rewards[::-1]: 
            R = r + 0.99 * R 
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards): 
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []


## Compression algorithm 

def compress_image(action): 
    """
    Apply the selected action (compression parameters) to compress the image. 
    """

    compressed_image = None 
    return compressed_image 

# Learned Compression: 
# Instead of using predefined compression parameters or techniques, 
# use a transformer-based model to learn the optimal way to compress images based on their features. 
# This could involve training a model to minimize a loss function that balances compression ratio and image quality, 
# possibly using a variant of the rate-distortion loss.

class ImageCompressionTransformer(nn.Module): 
    def __init__(self, feature_dim, num_heads, num_encoder_layers, num_decoder_layers):
        super(ImageCompressionTransformer, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        for param in self.feature_extractor.parameters(): 
            param.requires_grad = False # Freeze feature extractor 

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = feature_dim, nhead=num_heads, num_layers = num_encoder_layers)
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, num_layers = num_decoder_layers)
        )
        self.output_layer = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # Extract features 
        features = self.feature_extractor(x)

        features = features.view(features.size(0), -1, features.size(1))

        # Encoder 
        encoded_features = self.encoder(features)

        # Decoder 
        decoded_features = self.decoder(encoded_features)

        # Output layer 
        output = self.output_layer(decoded_features)
        return output 


def calculate_reward(original_image, compressed_image): 
    """
    Calculate the reward based on the original and compressed images. 
    This could involve metrics like compression ratio, PSNR, SSIM, etc. 
    """
    # Placeholder: Implement reward calculation 
    reward = 0
    return reward 





# Training 
# - During training, the agent interacts with the environment by compressing images based on its current policy 
# - Receives a reward based on the quality and size of the compressed image 
# - Updates its policy to maximize future rewards 

def train_agent(agent, image_paths, epochs = 10): 
    for epoch in range(epochs): 
        total_reward = 0 
        for image_path in image_paths: 
            # Preprocess and extract features 
            image_tensor = preprocess_image(image_path)
            features = extract_features(image_tensor).detach().numpy().flatten()

            # Select an action 
            action = agent.select_action(features)

            # Placeholder: Compress the image based on the selected action 
            compressed_image = compress_image(action)

            # Placeholder: Calculate the reward 
            original_image = Image.open(image_path)
            reward = calculate_reward(original_image, compressed_image)

            # Update the agent's policy 
            agent.rewards.append(reward)
            total_reward += reward 
        
        # Update the policy after each epoch 
        agent.update_policy()
        print(f"Epoch {epoch+1}/{epochs}, Total Reward: {total_reward}")

# Example usage 
image_folder = 'path/to/your/images'
image_paths = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]
agent = RLAgent(input_size = 512, hidden_size = 256, output_size = 10, learning_rate=1e-3) # Adjust sizes accordingly 
train_agent(agent, image_paths)

# Evaluation 
# - Regularly evaluate the agent on a set of test images not seen during training to measure the performance of the compression scheme. 
# - Adjust the reward function or the RL algorithm's parameters as needed based on performance. 

def evaluate_agent(agent, image_paths): 
    total_reward = 0 
    for image_path in image_paths: 
        # Preprocess and extract features 
        image_tensor = preprocess_image(image_path)
        features = extract_features(image_tensor).detach().numpy().flatten()

        # Select an action based on the current policy 
        action = agent.select_action(features)

        # Compress the image based on the selected action 
        compressed_image = compress_image(action)

        # Calculate the reward 
        original_image = Image.open(image_path)
        reward = calculate_reward(original_image, compressed_image)

        # Accumulate the total reward 
        total_reward += reward 

    average_reward = total_reward / len(image_paths)
    print(f"Average Reward: {average_reward}")
    return average_reward 

# Example usage 
test_image_folder = 'path/to/your/test/images'
test_image_paths = [os.path.join(test_image_folder, image) for image in os.listdir(test_image_folder)]
evaluate_agent(agent, test_image_paths)


### Action Space: 
# a bunch of transfomer and mamba models with different parameters trained.  
# Reward function: should be a tradeoff of how good the result/decoding image is and how much compute you used. 
# This is good because the model automatically learns which features deserve more compute and which features deserve less compute. 

# Extension: have an RL agent that automatically trains the optimal compute of the transformer/pixel-pixel image transformation. 
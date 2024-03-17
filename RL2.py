import numpy as np 
from keras.applications.vgg16 import VGG16, preprocess_input 
from keras.preprocessing import image 
import random 

## Coding the encoder 
## Goal: given any image, spit out the most optimal way using 0s and 1s with the shortest amount of bits to encode the image 

model = VGG16(weights = 'imagenet', include_top = False)

def get_image_features(img_path):
    img = image.load_img(img_path, target_size=[224, 224])
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expanded_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(img_array_expanded_dims)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    return flattened_features 

class LabelerEnvironment: 
    def __init__(self, img_paths): 
        self.img_paths = img_paths 
        self.state = None 
        self.beta = 0.1 # Penalty coefficient 
        self.done = False 
        self.reset() 

    def get_new_state(self): 
        img_path = random.choice(self.img_paths)
        return get_image_features(img_path)

    def reset(self): 
        self.state = self.get_new_state()
        self.done = False 
        return self.state 
    
    def step(self, action):  ## this part needs more fixing 
        if action == 2: # Assuming 2 represents 'STOP' 
            self.done = True 
            reward = 0 # No reward for stopping 
        else: 
            reward = -self.beta # Penalty for an action, you can adjust based on  checker's loss 
        next_state = self.get_new_state() # Get new state regardless of action 
        return next_state, reward, self.done 
    

class SimpleLabelerAgent: 
    def __init__(self, action_size, state_size, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01): 
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate 
        self.discount_rate = discount_rate 
        self.exploration_rate = exploration_rate 
        self.exploration_decay = exploration_decay 
        self.min_exploration_rate = min_exploration_rate 
        self.action_size = action_size 

    def choose_action(self, state_index): 
        if np.random.rand() < self.exploration_rate: 
            return np.random.choice([0, 1, 2]) # Explore 
        else: 
            return np.argmax(self.q_table[state_index]) # Exploit learned values 
        
    def update_q_table(self, state_index, action, reward, next_state_index, done): 
        if not done: 
            max_future_q = np.max(self.q_table[next_state_index])
            current_q = self.q_table[state_index, action]
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_rate * max_future_q) 
            self.q_table[state_index, action] = new_q 
        else: 
            self.q_table[state_index, action] = (1 - self.learning_rate) * self.q_table[state_index, action] + self.learning_rate * reward 
        
        # Update exploration  rate 
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

def main(): 
    img_paths = [ ' fill this in ']
    env = LabelerEnvironment(img_paths) 
    agent = SimpleLabelerAgent(action_size = 3, state_size = env.state.size, learning_rate = 0.1, discount_rate = 0.95, exploration_rate = 1.0, exploration_decay = 0.99, min_exploration_rate = 0.01) 

    episodes = 10 
    for episode in range(episodes): 
        state = env.reset() ## get state (vector representation of some image)
        total_reward = 0 
        done = False 
        while not done: 
            # For simplicity, using the state directly. In practice, you might need to map it to a suitable representation 
            state_index = 0 # Simplification, replace with actual state handling 
            action = agent.choose_action(state_index)
            next_state, reward, done = env.step(action)
            next_state_index = 0 # Simplification, replace with actual state handling 
            agent.update_q_table(state_index, action, reward, next_state_index, done) 
            total_reward += reward 
            state = next_state 

        print(f"Episode {episode+1}: Total reward = {total_reward}") 


if __name__ == "__main__":
    main() 


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from cnn_dqn import GridWorld, ReplayBuffer

class FrameStackCnnDQN(nn.Module):
    def __init__(self, input_shape=(4, 11, 11), num_actions=8):
        super(FrameStackCnnDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 11x11 -> 5x5 -> 2x2. 32 * 2 * 2 = 128
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_framestack():
    device = torch.device("cpu")
    env = GridWorld(size=15, obstacle_ratio=0.18, max_steps=200, seed=42, obs_radius=5)
    model = FrameStackCnnDQN().to(device)
    target_model = FrameStackCnnDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    memory = ReplayBuffer(capacity=10000)
    
    batch_size = 64
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    target_update = 10
    
    num_episodes = 250 # train for 250 episodes for speed to get a quick evaluation
    
    epsilon = epsilon_start
    best_reward = -float('inf')
    
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        # Initial frame stack: copy the first observation 4 times
        frames = deque([obs]*4, maxlen=4)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            state = np.array(frames)
            
            if random.random() < epsilon:
                action = random.randrange(8)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = model(state_t)
                action = q_vals.argmax().item()
                
            next_obs, reward, done, _ = env.step(action)
            frames.append(next_obs)
            next_state = np.array(frames)
            
            state_t_push = torch.FloatTensor(state)
            next_state_t_push = torch.FloatTensor(next_state)
            memory.push(state_t_push, action, reward, next_state_t_push, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                
                states = batch.state.to(device)
                actions = batch.action.to(device)
                rewards = batch.reward.to(device)
                next_states = batch.next_state.to(device)
                dones = batch.done.float().to(device)
                
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if ep % target_update == 0:
            target_model.load_state_dict(model.state_dict())
            
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), 'framestack_best.pth')
            
        if ep % 50 == 0:
            print(f"Ep {ep}/{num_episodes}, Steps: {steps}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    print("Training complete. Evaluating on 100 test environments...")
    
    # Evaluation
    model.load_state_dict(torch.load('framestack_best.pth', weights_only=True))
    model.eval()
    
    test_rewards = []
    test_paths = []
    successes = 0
    
    for i in range(100):
        test_env = GridWorld(size=15, obstacle_ratio=0.18, max_steps=200, seed=1000 + i, obs_radius=5)
        obs = test_env.reset()
        frames = deque([obs]*4, maxlen=4)
        done = False
        steps = 0
        r_sum = 0
        
        while not done:
            state_t = torch.FloatTensor(np.array(frames)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_t).argmax().item()
            obs, reward, done, _ = test_env.step(action)
            frames.append(obs)
            r_sum += reward
            steps += 1
        
        test_rewards.append(r_sum)
        test_paths.append(steps)
        if test_env.grid[test_env.agent] == test_env.grid[test_env.goal]:
            successes += 1
            
    print(f"Test SR: {successes}%")
    print(f"Test Path: {np.mean(test_paths):.2f} +/- {np.std(test_paths):.2f}")
    
if __name__ == '__main__':
    train_framestack()

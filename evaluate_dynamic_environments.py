import numpy as np
import torch
import random
from pathlib import Path
import csv

from cnn_dqn import GridWorld, CnnDQN, DuelingCnnLstmDQN, Device

class DynamicGridWorld(GridWorld):
    def __init__(self, size=15, obstacle_ratio=0.18, max_steps=200, seed=100, obs_radius=None, k_move=3, dyn_ratio=0.25):
        super().__init__(size=size, obstacle_ratio=obstacle_ratio, max_steps=max_steps, seed=seed, obs_radius=obs_radius)
        self.k_move = k_move
        self.dyn_ratio = dyn_ratio
        self.dyn_rng = random.Random(seed)
        self._init_obstacles()
        
    def _init_obstacles(self):
        # find all obstacle coordinates (val == 1.0)
        self.obstacles = [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r, c] == 1.0]
        
        if len(self.obstacles) == 0:
            self.num_dyn = 0
            self.dyn_indices = []
        else:
            self.num_dyn = max(1, int(len(self.obstacles) * self.dyn_ratio))
            self.dyn_indices = self.dyn_rng.sample(range(len(self.obstacles)), self.num_dyn)
        
    def reset(self):
        obs = super().reset()
        self._init_obstacles()
        return obs

    def step(self, action: int):
        # 1. Standard agent step
        obs, reward, done, info = super().step(action)
        
        # 2. Every k_move steps, move the dynamic obstacles if not done
        if not done and (self.steps % self.k_move == 0):
            self._move_obstacles()
            # Update observation after obstacle motion
            obs = self._obs()
            
            # Check if an obstacle moved directly onto the robot
            if self.grid[self.agent] == 1.0:
                reward -= 0.3
                info['collision_dynamic'] = True
                
        return obs, reward, done, info

    def _move_obstacles(self):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for idx in self.dyn_indices:
            curr_r, curr_c = self.obstacles[idx]
            # Try a random move
            self.dyn_rng.shuffle(moves)
            for dr, dc in moves:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    # Valid move if free cell, not agent, not goal
                    if self.grid[nr, nc] == 0.0 and (nr, nc) != self.agent and (nr, nc) != self.goal:
                        # Move obstacle
                        self.grid[curr_r, curr_c] = 0.0
                        self.grid[nr, nc] = 1.0
                        self.obstacles[idx] = (nr, nc)
                        break

def evaluate_dynamic(model, algo, obs_radius, k_move=3, num_ep=100, seed=100):
    model.eval()
    use_lstm = (algo == 'lstm')
    metrics = []
    
    for ep in range(num_ep):
        env = DynamicGridWorld(size=15, obstacle_ratio=0.18, max_steps=200, seed=seed + ep, obs_radius=obs_radius, k_move=k_move)
        state_np = env.reset()
        state = torch.from_numpy(state_np).unsqueeze(0).unsqueeze(0).to(Device)
        ep_rew = 0.0
        seq = []
        hidden = None
        
        for step_num in range(200):
            with torch.no_grad():
                if use_lstm:
                    frame = state.squeeze(0)
                    if len(seq) < 4:
                        seq.append(frame)
                        while len(seq) < 4:
                            seq.insert(0, seq[0])
                    else:
                        seq.pop(0)
                        seq.append(frame)
                    seq_t = torch.stack(seq, dim=0).unsqueeze(0)
                    q, hidden = model(seq_t, hidden)
                    act = q[:, -1, :].argmax(dim=1).item()
                else:
                    q = model(state)
                    act = q.argmax(dim=1).item()
                    
            next_state_np, rew, done, info = env.step(act)
            state = torch.from_numpy(next_state_np).unsqueeze(0).unsqueeze(0).to(Device)
            ep_rew += rew
            if done:
                break
                
        metrics.append({
            'episode': ep + 1,
            'steps': step_num + 1,
            'reward': ep_rew,
            'reached_goal': int(info.get('reached_goal', False))
        })
        
    srs = [m['reached_goal'] for m in metrics]
    paths = [m['steps'] for m in metrics]
    rewards = [m['reward'] for m in metrics]
    return np.mean(srs)*100, np.mean(paths), np.std(paths, ddof=1), np.mean(rewards), np.std(rewards, ddof=1)

if __name__ == '__main__':
    base_path = 'github_repository/checkpoints/baseline_cnn_dqn.pth'
    lstm_path = 'github_repository/checkpoints/lstm_cnn_dqn.pth'
    
    m_base = CnnDQN(15, 8).to(Device)
    m_base.load_state_dict(torch.load(base_path, map_location=Device))
    
    m_lstm = DuelingCnnLstmDQN(11, 8, lstm_hidden=128, sequence_length=4).to(Device)
    m_lstm.load_state_dict(torch.load(lstm_path, map_location=Device))
    
    print('='*75)
    print('DYNAMIC ENVIRONMENT EVALUATION (Non-Stationary Obstacles, k=2, 3, 5)')
    print('='*75)
    for k in [5, 3, 2]:
        sr_b, p_b, ps_b, r_b, rs_b = evaluate_dynamic(m_base, 'baseline', None, k_move=k)
        sr_l, p_l, ps_l, r_l, rs_l = evaluate_dynamic(m_lstm, 'lstm', 5, k_move=k)
        print(f'Interval k={k} steps:')
        print(f'  Baseline CNN-DQN : SR={sr_b:5.1f}% | Path={p_b:6.2f} +/- {ps_b:6.2f} | Reward={r_b:6.2f} +/- {rs_b:6.2f}')
        print(f'  LSTM-CNN-DQN     : SR={sr_l:5.1f}% | Path={p_l:6.2f} +/- {ps_l:6.2f} | Reward={r_l:6.2f} +/- {rs_l:6.2f}')
        print(f'  Advantage (LSTM) : SR Delta=+{sr_l - sr_b:4.1f}% | Path Reduction={p_b - p_l:5.2f} steps\n')

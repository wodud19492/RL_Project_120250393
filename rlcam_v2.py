import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):

    def __init__(self, in_channels, n_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class GridMaskEnv:
    def __init__(self, model, input_tensor, item_idx, target_class, grid_size=(8, 10), max_batch_steps=5):
        self.model = model
        self.input_tensor = input_tensor
        self.item_idx = item_idx
        self.target_class = target_class
        self.grid_size = grid_size
        self.max_batch_steps = max_batch_steps
        
        self.n_actions = grid_size[0] * grid_size[1]
        self.device = input_tensor.device
        
        _, _, self.H, self.W = input_tensor.shape
        self.min_val = self.input_tensor.min().item()

        self.valid_patches = self._compute_valid_patches()

        self.reset()

    def _compute_valid_patches(self):
        valid_mask = torch.zeros(self.n_actions, device=self.device)
        
        patch_h = self.H // self.grid_size[0]
        patch_w = self.W // self.grid_size[1]
        
        for idx in range(self.n_actions):
            r = idx // self.grid_size[1]
            c = idx % self.grid_size[1]
            
            r_start, r_end = r * patch_h, (r + 1) * patch_h
            c_start, c_end = c * patch_w, (c + 1) * patch_w
            
            patch_area = self.input_tensor[0, 0, r_start:r_end, c_start:c_end]
            patch_mean = patch_area.mean().item()
            
            if patch_mean > self.min_val + 5.0:
                valid_mask[idx] = 1.0
                
        return valid_mask

    def reset(self):
        self.mask_grid = torch.zeros(self.n_actions, device=self.device)
        
        self.steps = 0
        state, mask_full = self._get_state()
        
        masked_input = self._apply_mask(self.input_tensor, mask_full)
        with torch.no_grad():
            logits = self.model(masked_input)
            probs = F.softmax(logits[0, self.item_idx], dim=0)
            self.current_score = probs[self.target_class].item()
            
        return state, mask_full

    def _get_state(self):
        mask_2d = self.mask_grid.view(self.grid_size[0], self.grid_size[1])
        mask_full = F.interpolate(
            mask_2d.unsqueeze(0).unsqueeze(0), 
            size=(self.H, self.W), 
            mode='nearest'
        ) 
        state = torch.cat([self.input_tensor, mask_full], dim=1)
        return state, mask_full

    def _apply_mask(self, img, mask):
        return img * mask + self.min_val * (1 - mask)

    def step(self, action_indices):
        if isinstance(action_indices, (int, np.integer)):
            action_indices = [action_indices]
        elif isinstance(action_indices, np.ndarray):
            action_indices = action_indices.tolist()

        valid_count = 0

        rows, cols = self.grid_size

        for idx in action_indices:
            if self.valid_patches[idx] == 0.0:
                continue

            if self.mask_grid[idx] == 0.0:
                self.mask_grid[idx] = 1.0
                valid_count += 1
        
            r = idx // cols
            c = idx % cols
            
            r_min = max(0, r - 1)
            r_max = min(rows, r + 1)
            c_min = max(0, c - 1)
            c_max = min(cols, c + 1)
            
            for nr in range(r_min, r_max):
                for nc in range(c_min, c_max):
                    n_idx = nr * cols + nc
                    if self.mask_grid[n_idx] == 0.0:
                        self.mask_grid[n_idx] = 1.0
        
        self.steps += 1
        
        state, mask_full = self._get_state()
        
        if valid_count == 0:
            reward = -0.1 
            done = True
            return (state, mask_full), reward, done
        
        masked_input = self._apply_mask(self.input_tensor, mask_full)
        with torch.no_grad():
            logits = self.model(masked_input)
            probs = F.softmax(logits[0, self.item_idx], dim=0)
            new_score = probs[self.target_class].item()

        score_diff = new_score - self.current_score
        
        if score_diff > 0:
            reward = score_diff * 100.0
        elif score_diff == 0:
            reward = -0.5
        else:
            reward = -1.0
            
        self.current_score = new_score

        done = False
        if self.steps >= self.max_batch_steps:
            done = True
        
        if torch.sum(self.mask_grid) >= self.n_actions * 0.95:
            done = True
        
        return (state, mask_full), reward, done

class DQNAgent:
    def __init__(self, in_channels, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.90
        self.lr = 0.001
        self.policy_net = QNetwork(in_channels, n_actions).to(device)
        self.target_net = QNetwork(in_channels, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=2000)

    def select_action(self, state, available_mask, top_k=5):
            if random.random() <= self.epsilon:
                available_indices = [i for i, val in enumerate(available_mask) if val == 0]
                if not available_indices: 
                    return []
                k = min(len(available_indices), top_k)
                return random.sample(available_indices, k)
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    q_values[0, available_mask.bool()] = -float('inf')

                    remaining_count = (available_mask == 0).sum().item()
                    k = min(int(remaining_count), top_k)
                
                    if k == 0: return []

                    _, top_indices = torch.topk(q_values, k, dim=1)
                    return top_indices[0].cpu().numpy().tolist()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.cat([b[0] for b in batch])
        action_batch = torch.tensor([[b[1]] for b in batch], device=self.device)
        reward_batch = torch.tensor([b[2] for b in batch], device=self.device)
        next_state_batch = torch.cat([b[3] for b in batch])
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)

        next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
        
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class GrbasRLCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, input_tensor, item_idx=0, target_class=None, 
                 episodes=50, grid_size=(16, 20), batch_actions=20): 
        
        device = input_tensor.device
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = torch.argmax(logits[0, item_idx]).item()
            
            org_probs = F.softmax(logits[0, item_idx], dim=0)
            org_score = org_probs[target_class].item()
            print(f"Target Class: {target_class} | Original Score: {org_score:.4f}")

        in_channels = input_tensor.shape[1]
        n_actions = grid_size[0] * grid_size[1]
        
        max_batch_steps = 30
        
        env = GridMaskEnv(self.model, input_tensor, item_idx, target_class, 
                          grid_size=grid_size, max_batch_steps=max_batch_steps)
                          
        agent = DQNAgent(in_channels, n_actions, device)

        best_mask = None
        max_score = -1.0
        
        for e in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            
            if env.current_score > max_score:
                max_score = env.current_score
                _, mask_full = env._get_state()
                best_mask = mask_full.detach().clone()
            
            while True:
                actions = agent.select_action(state, env.mask_grid, top_k=batch_actions)
                if not actions: break
                
                (next_state, mask_full), reward, done = env.step(actions)
                
                for action_idx in actions:
                    agent.store_transition(state, action_idx, reward, next_state, done)
                agent.train_step()
                
                state = next_state
                total_reward += reward
                
                if env.current_score > max_score:
                    max_score = env.current_score
                    best_mask = mask_full.detach().clone()

                if done:
                    break
            
            agent.update_target_network()
            agent.decay_epsilon()
            
            if (e+1) % 5 == 0:
                print(f"Ep {e+1} | Score: {env.current_score:.4f} (Peak: {max_score:.4f}) | R: {total_reward:.2f}")

        if best_mask is None:
            return torch.ones_like(input_tensor[0,0]).cpu().numpy(), target_class

        final_mask = best_mask[0, 0].cpu().numpy()
        return final_mask, target_class

    def load_agent(self, checkpoint_path, in_channels, n_actions, device):

        agent = DQNAgent(in_channels, n_actions, device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.policy_net.load_state_dict(ckpt['model_state_dict'])
        agent.target_net.load_state_dict(ckpt['model_state_dict'])
        
        agent.epsilon = 0.0
        agent.epsilon_min = 0.0
        agent.policy_net.eval()
        return agent

    def generate_with_trained_agent(self, input_tensor, agent_ckpt_path, 
                                    item_idx=0, grid_size=(16, 20), batch_actions=20):
        device = input_tensor.device
        in_channels = input_tensor.shape[1]
        n_actions = grid_size[0] * grid_size[1]
        
        if not hasattr(self, 'trained_agent'):
             self.trained_agent = self.load_agent(agent_ckpt_path, in_channels, n_actions, device)
        
        agent = self.trained_agent

        with torch.no_grad():
            logits = self.model(input_tensor)
            target_class = torch.argmax(logits[0, item_idx]).item()

        env = GridMaskEnv(self.model, input_tensor, item_idx, target_class, 
                          grid_size=grid_size, max_batch_steps=6)

        state, _ = env.reset()
        best_mask = None
        max_score = -1.0
        
        while True:
            actions = agent.select_action(state, env.mask_grid, top_k=batch_actions)
            if not actions: break
            
            (next_state, mask_full), reward, done = env.step(actions)
            state = next_state
            
            if env.current_score > max_score:
                max_score = env.current_score
                best_mask = mask_full.detach().clone()
            
            if done:
                break
                
        if best_mask is None:
            return torch.ones_like(input_tensor[0,0]).cpu().numpy(), target_class

        final_mask = best_mask[0, 0].cpu().numpy()
        return final_mask, target_class
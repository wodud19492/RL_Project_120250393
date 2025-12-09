import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from models_v3 import AutoGRBASModel, PatientGrbasDistDataset
from rlcam_v2 import DQNAgent, GridMaskEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_crnn_5587.pt"
TRAIN_CSV = "GRBAS_train.csv"
SAVE_PATH = "best_rl_agent2.pth"

GRID_SIZE = (80, 100)
BATCH_ACTIONS = 100
EPOCHS = 5
EPISODES_PER_IMG = 5

def main():
    print(f"Loading Environment Model from {CKPT_PATH}...")
    
    env_model = AutoGRBASModel(in_channels=3).to(DEVICE)
    
    dummy = torch.zeros(1, 3, 80, 100).to(DEVICE)
    env_model.eval()
    env_model.encoder(dummy)
    
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    env_model.load_state_dict(ckpt["model_state_dict"])
    
    for param in env_model.parameters():
        param.requires_grad = False

    print(f"Loading Training Data from {TRAIN_CSV}...")

    dataset = PatientGrbasDistDataset(csv_path=TRAIN_CSV)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    in_channels = 3 
    n_actions = GRID_SIZE[0] * GRID_SIZE[1]
    
    agent = DQNAgent(in_channels, n_actions, DEVICE)
    
    agent.epsilon = 1.0
    agent.epsilon_decay = 0.9999 
    agent.epsilon_min = 0.1

    print("Start Global RL Training...")
    
    global_step = 0
    
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_reward = 0
        count = 0
        
        for batch_idx, sample in enumerate(pbar):
            input_tensor = sample['feat'].to(DEVICE)
            
            with torch.no_grad():
                logits = env_model(input_tensor)
                target_class = torch.argmax(logits[0, 0]).item()

            env = GridMaskEnv(env_model, input_tensor, item_idx=0, target_class=target_class, 
                              grid_size=GRID_SIZE, max_batch_steps=6)

            for ep in range(EPISODES_PER_IMG):
                state, _ = env.reset()
                episode_reward = 0
                
                while True:
                    actions = agent.select_action(state, env.mask_grid, top_k=BATCH_ACTIONS)
                    if not actions: break
                    
                    (next_state, _), reward, done = env.step(actions)
                    
                    for action_idx in actions:
                        agent.store_transition(state, action_idx, reward, next_state, done)
                    
                    agent.train_step()
                    
                    state = next_state
                    episode_reward += reward
                    global_step += 1
                    
                    if done:
                        break
                
                if global_step % 100 == 0:
                    agent.update_target_network()
                
                agent.decay_epsilon()
                epoch_reward += episode_reward
                count += 1

            if (batch_idx + 1) % 10 == 0:
                avg_r = epoch_reward / count if count > 0 else 0
                pbar.set_postfix({
                    "Avg Reward": f"{avg_r:.2f}",
                    "Epsilon": f"{agent.epsilon:.2f}"
                })

        print(f"Saving Agent checkpoint to {SAVE_PATH}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }, SAVE_PATH)

    print("Training Complete!")

if __name__ == "__main__":
    main()
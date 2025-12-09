import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# 기존 모듈 임포트
from models_v3 import AutoGRBASModel, PatientGrbasDistDataset
from gradcam import GrbasGradCAM
from rlcam_v2 import DQNAgent, GridMaskEnv

# ==========================================
# 설정 (Configuration)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_CKPT_PATH = "best_crnn_5587.pt"
AGENT_CKPT_PATH = "best_rl_agent2.pth"
CSV_PATH = "input/sample.csv"
SAVE_DIR = "sample_results"

ITEMS = ["G", "R", "B", "A", "S"]
GRID_SIZE = (80, 100)
BATCH_ACTIONS = 100

def load_env_model():
    print(f"Loading Environment Model from {ENV_CKPT_PATH}...")
    model = AutoGRBASModel(in_channels=3).to(DEVICE)

    dummy = torch.zeros(1, 3, 80, 100).to(DEVICE)
    model.eval()
    model.encoder(dummy)
    
    ckpt = torch.load(ENV_CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def load_trained_agent(n_actions):
    print(f"Loading Trained Agent from {AGENT_CKPT_PATH}...")

    agent = DQNAgent(in_channels=3, n_actions=n_actions, device=DEVICE)
    
    ckpt = torch.load(AGENT_CKPT_PATH, map_location=DEVICE)
    agent.policy_net.load_state_dict(ckpt['model_state_dict'])

    agent.policy_net.eval()
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0
    
    return agent

def save_comparison_image(spectrogram, grad_mask, rl_mask, title, save_path):

    if rl_mask.max() - rl_mask.min() < 1e-6:
        rl_mask = np.zeros_like(rl_mask)
    else:
        rl_mask = (rl_mask - rl_mask.min()) / (rl_mask.max() - rl_mask.min())
        rl_mask = cv2.GaussianBlur(rl_mask, (31, 31), 11)
        if rl_mask.max() > 0:
            rl_mask = rl_mask / rl_mask.max()

    plt.figure(figsize=(10, 9))
    n_freq, n_frames = spectrogram.shape
    extent = [0, (n_frames * 256) / 16000, 0, n_freq]
    
    plt.subplot(3, 1, 1)
    plt.title(f"[Original] {title}")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno', extent=extent)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("Mel Bin")
    plt.xticks([])
    
    plt.subplot(3, 1, 2)
    plt.title("[Grad-CAM] Baseline")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(grad_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Bin")
    plt.xticks([])

    plt.subplot(3, 1, 3)
    plt.title("[RL-CAM] Ours")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(rl_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Bin")
    plt.xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def set_hybrid_mode(model):
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    env_model = load_env_model()
    n_actions = GRID_SIZE[0] * GRID_SIZE[1]
    agent = load_trained_agent(n_actions)

    ds = PatientGrbasDistDataset(csv_path=CSV_PATH)

    target_layer = env_model.encoder.cnn[2].net[0]
    grad_runner = GrbasGradCAM(env_model, target_layer)
    
    print(f"Starting Inference on {len(ds)} patients...")
    
    for i in tqdm(range(len(ds)), desc="Testing"):
        try:
            sample = ds[i]
            pid = sample['patient_id']

            input_tensor = sample['feat'].unsqueeze(0).to(DEVICE)
            input_tensor.requires_grad = True

            spectrogram = input_tensor[0, 0].detach().cpu().numpy()
            
            for item_idx, item_name in enumerate(ITEMS):

                set_hybrid_mode(env_model)
                grad_mask, pred_cls_idx = grad_runner(input_tensor, item_idx=item_idx)
                env_model.eval()

                env = GridMaskEnv(env_model, input_tensor, item_idx=item_idx, 
                                  target_class=pred_cls_idx, grid_size=GRID_SIZE, max_batch_steps=6)

                state, _ = env.reset()
                best_mask = None
                max_score = -1.0

                while True:
                    actions = agent.select_action(state, env.mask_grid, top_k=BATCH_ACTIONS)
                    if not actions: break

                    (next_state, mask_full), _, done = env.step(actions)

                    state = next_state
                    
                    if env.current_score > max_score:
                        max_score = env.current_score
                        best_mask = mask_full.detach().clone()
                    
                    if done:
                        break
                
                if best_mask is None:
                    rl_mask = np.zeros_like(spectrogram)
                else:
                    rl_mask = best_mask[0, 0].cpu().numpy()

                score = pred_cls_idx + 1
                filename = f"{pid}_{item_name}_Score{score}.png"
                save_path = os.path.join(SAVE_DIR, filename)
                
                save_comparison_image(
                    spectrogram, grad_mask, rl_mask,
                    title=f"Patient {pid} | {item_name}",
                    save_path=save_path
                )
                
        except Exception as e:
            print(f"Error on patient {i}: {e}")
            continue

    print(f"Results saved in '{SAVE_DIR}'")

if __name__ == "__main__":
    main()
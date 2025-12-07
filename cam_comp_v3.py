import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from models_v3 import AutoGRBASModel, PatientGrbasDistDataset
from rlcam_v3 import GrbasRLCAM
from gradcam import GrbasGradCAM

# ==========================================
# 설정
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_crnn_5587.pt"
CSV_PATH = "GRBAS_test.csv"
SAVE_DIR = "results_crnn_3vowel2" # 저장 폴더 변경
ITEMS = ["G", "R", "B", "A", "S"]
VOWELS = ["a", "i", "u"]         # 채널 순서대로 매핑

# RL-CAM 파라미터
STEPS = 100
LAMBDA_REG = 0.05 

# ==========================================
# 시각화 함수
# ==========================================
def save_comparison_image(spectrogram, grad_mask, rl_mask, title, save_path):
    plt.figure(figsize=(10, 9))
    
    n_freq, n_frames = spectrogram.shape
    sample_rate = 16000
    hop_length = 256
    duration = (n_frames * hop_length) / sample_rate
    
    extent = [0, duration, 0, n_freq]
    
    # 1. Original
    plt.subplot(3, 1, 1)
    plt.title(f"[Original] {title}")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno', extent=extent)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("Mel Frequency Bin")
    plt.xticks([])
    
    # 2. Grad-CAM
    plt.subplot(3, 1, 2)
    plt.title("[Grad-CAM] Baseline")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(grad_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Frequency Bin")
    plt.xticks([])

    # 3. RL-CAM
    plt.subplot(3, 1, 3)
    plt.title("[RL-CAM] Ours (Optimized)")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(rl_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Frequency Bin")
    plt.xlabel("Time (seconds)")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ==========================================
# Main
# ==========================================
def load_model():
    print(f"Loading CRNN Model from {CKPT_PATH}...")
    model = AutoGRBASModel(in_channels=3).to(DEVICE)
    dummy = torch.zeros(1, 3, 80, 100).to(DEVICE)
    model.eval()
    model.encoder(dummy)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    model = load_model()
    ds = PatientGrbasDistDataset(csv_path=CSV_PATH)
    
    # Runners
    target_layer = model.encoder.cnn[2].net[0]
    grad_runner = GrbasGradCAM(model, target_layer)
    rl_runner = GrbasRLCAM(model)
    
    print(f"Starting 3-Vowel Analysis for {len(ds)} patients...")
    
    # 편의상 10명만 테스트
    for i in tqdm(range(min(len(ds), 10)), desc="Processing"):
        try:
            sample = ds[i]
            pid = sample['patient_id']
            
            # (1, 3, F, T) - 3개 채널 포함
            input_tensor = sample['feat'].unsqueeze(0).to(DEVICE)
            input_tensor.requires_grad = True
            
            for item_idx, item_name in enumerate(ITEMS):
                
                # 1. Grad-CAM 생성 (3채널 통합 Gradient)
                model.train()
                for m in model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
                        m.eval()
                grad_mask, pred_cls_idx = grad_runner(input_tensor, item_idx=item_idx)
                
                # 2. RL-CAM 생성 (3채널 통합 Mask)
                rl_mask, final_score = rl_runner.generate(
                    input_tensor,
                    item_idx=item_idx,
                    steps=STEPS,
                    lr=0.5,
                    lambda_reg=LAMBDA_REG
                )
                
                # 3. [핵심] 3개 모음 각각 저장하기
                # 마스크(grad_mask, rl_mask)는 공유하지만, 
                # 배경이 되는 스펙트로그램(spectrogram)을 채널별로 바꿔가며 저장
                for v_idx, v_name in enumerate(VOWELS): # 0:a, 1:i, 2:u
                    
                    # 해당 채널의 스펙트로그램 추출
                    spectrogram = input_tensor[0, v_idx].detach().cpu().numpy()
                    
                    # 파일명: P001_G_Score3.5_a.png
                    filename = f"{pid}_{item_name}_Score{final_score:.1f}_{v_name}.png"
                    save_path = os.path.join(SAVE_DIR, filename)
                    
                    save_comparison_image(
                        spectrogram, grad_mask, rl_mask,
                        title=f"Patient {pid} | {item_name} (Pred: {final_score:.2f}) | /{v_name}/",
                        save_path=save_path
                    )
                
        except Exception as e:
            print(f"Error on {i}: {e}")
            continue

    print(f"Done! Check '{SAVE_DIR}'")

if __name__ == "__main__":
    main()
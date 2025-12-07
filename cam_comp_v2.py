import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from models_v3 import AutoGRBASModel, PatientGrbasDistDataset
#from rlcam import GrbasRLCAM
from gradcam import GrbasGradCAM # 기존 Grad-CAM도 비교용으로 사용
from rlcam_v2 import GrbasRLCAM
# ==========================================
# 설정
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_crnn_5587.pt"
CSV_PATH = "GRBAS_test.csv"
SAVE_DIR = "results_crnn_vis2"
ITEMS = ["G", "R", "B", "A", "S"]

# RL-CAM 파라미터 (CNN에서 검증된 값 적용)
STEPS = 100
LAMBDA_REG = 0.05

# ==========================================
# 시각화 함수 (시간축 적용)
# ==========================================
def save_comparison_image(spectrogram, grad_mask, rl_mask, title, save_path):
    plt.figure(figsize=(10, 9))
    
    # 시간 축 계산 (Frame -> Seconds)
    n_freq, n_frames = spectrogram.shape
    sample_rate = 16000
    hop_length = 256
    duration = (n_frames * hop_length) / sample_rate
    
    # extent = [left, right, bottom, top]
    extent = [0, duration, 0, n_freq]
    
    # 1. Original
    plt.subplot(3, 1, 1)
    plt.title(f"[Original] {title}")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno', extent=extent)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("Mel Frequency Bin")
    plt.xticks([]) # x축 눈금 숨김
    
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
    plt.xlabel("Time (seconds)") # 시간 단위 표시
    
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
    model.encoder(dummy) # Lazy init
    
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    model = load_model()
    ds = PatientGrbasDistDataset(csv_path=CSV_PATH)
    
    # Grad-CAM Runner
    target_layer = model.encoder.cnn[2].net[0]
    grad_runner = GrbasGradCAM(model, target_layer)
    
    # RL-CAM Runner
    rl_runner = GrbasRLCAM(model)
    
    print(f"Starting CRNN Visualization for {len(ds)} patients...")
    
    # 20명만 샘플링
    for i in tqdm(range(min(len(ds), 20)), desc="Processing"):
        try:
            sample = ds[i]
            pid = sample['patient_id']
            
            # CRNN 입력: (1, 3, F, T)
            input_tensor = sample['feat'].unsqueeze(0).to(DEVICE)
            input_tensor.requires_grad = True
            
            # 시각화는 0번 채널(/a/) 기준
            spectrogram = input_tensor[0, 0].detach().cpu().numpy()
            
            for item_idx, item_name in enumerate(ITEMS):
                
                # 1. Grad-CAM (Hybrid Mode 수동 설정)
                model.train()
                for m in model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
                        m.eval()
                
                grad_mask, pred_cls_idx = grad_runner(input_tensor, item_idx=item_idx)
                
                # 2. RL-CAM (Generate 내부에서 모드 자동 설정)
                # target_class는 사용하지 않음 (Preserve Score 방식이므로)
                rl_mask, final_score = rl_runner.generate(
                    input_tensor,
                    item_idx=item_idx,
                    steps=STEPS,
                    lr=0.5,
                    lambda_reg=LAMBDA_REG
                )
                
                filename = f"{pid}_{item_name}_Score{final_score:.1f}.png"
                save_path = os.path.join(SAVE_DIR, filename)
                
                save_comparison_image(
                    spectrogram, grad_mask, rl_mask,
                    title=f"Patient {pid} | {item_name} (Pred: {final_score:.2f})",
                    save_path=save_path
                )
                
        except Exception as e:
            print(f"Error on {i}: {e}")
            continue

    print(f"Done! Check '{SAVE_DIR}'")

if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# 모델 및 데이터셋 임포트
from models_v3 import GrbasCNN, GrbasDataset
from rlcam_cnn import GrbasCnnRLCAM

# ==========================================
# 설정
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_cnn.pt"  # CNN 모델 체크포인트
CSV_PATH = "GRBAS_test.csv"
SAVE_DIR = "results_cnn_comparison_5"
ITEMS = ["G", "R", "B", "A", "S"]
STEPS = 100
LAMBDA_REG = 0.15

# ==========================================
# CNN 전용 Grad-CAM 클래스 (내장)
# ==========================================
class CnnGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # 타겟 레이어: 마지막 Conv Layer
        # GrbasCNN의 conv_block은 Sequential이므로 뒤에서부터 Conv2d 탐색
        last_block = self.model.features[-1]
        target_layer = last_block.net[0]
                
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, feat, vowel_id, item_idx):
        # 1. Forward
        self.model.zero_grad()
        output = self.model(feat, vowel_id) # (1, 5)
        
        # 2. Target Score
        score = output[0, item_idx]
        
        # 3. Backward
        score.backward(retain_graph=True)
        
        # 4. CAM Calculation
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of Gradients
        weights = torch.mean(gradients, dim=(2, 3))[0] # (Ch,)
        
        # Weighted Sum
        cam = torch.zeros(activations.shape[2:], device=feat.device)
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]
            
        cam = torch.nn.functional.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Normalize
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
        # Resize to Input Size
        h, w = feat.shape[2], feat.shape[3]
        cam = cv2.resize(cam, (w, h))
        
        return cam, score.item()

# ==========================================
# 유틸리티 함수
# ==========================================
def load_cnn_model():
    print(f"Loading CNN Model from {CKPT_PATH}...")
    # GrbasCNN 생성 (Vowel Embedding 사용 여부 주의)
    model = GrbasCNN(num_vowels=3, use_vowel_embedding=True).to(DEVICE)
    
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    return model

def save_comparison_image(spectrogram, grad_mask, rl_mask, title, save_path):
    plt.figure(figsize=(10, 9))
    
    # 1. 시간 축 계산 (Frame -> Seconds)
    # Spectrogram Shape: (Freq, Time)
    n_freq, n_frames = spectrogram.shape
    
    sample_rate = 16000  # 모델 기본값
    hop_length = 256     # 모델 기본값
    
    duration = (n_frames * hop_length) / sample_rate
    
    # extent = [x_min, x_max, y_min, y_max]
    # x축: 0초 ~ duration초, y축: 0 ~ 80 (Mel Bin)
    extent = [0, duration, 0, n_freq]
    
    # -------------------------------------------------------
    
    # 1. Original
    plt.subplot(3, 1, 1)
    plt.title(f"[Original] {title}")
    # [수정] extent 추가, xlabel 변경
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno', extent=extent)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("Mel Frequency Bin")
    plt.xticks([]) # 맨 위 그래프는 x축 눈금 생략 (깔끔하게)
    
    # 2. Grad-CAM
    plt.subplot(3, 1, 2)
    plt.title("[Grad-CAM] Baseline")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(grad_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Frequency Bin")
    plt.xticks([]) # 중간 그래프도 x축 눈금 생략

    # 3. RL-CAM
    plt.subplot(3, 1, 3)
    plt.title("[RL-CAM] Ours (Optimized)")
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=0.5, extent=extent)
    plt.imshow(rl_mask, aspect='auto', origin='lower', cmap='jet', alpha=0.6, extent=extent)
    plt.colorbar()
    plt.ylabel("Mel Frequency Bin")
    
    # [수정] x축 라벨을 'Time Frame'에서 'Time (seconds)'로 변경
    plt.xlabel("Time (seconds)")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
# ==========================================
# Main
# ==========================================
def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    model = load_cnn_model()
    
    # CNN은 일반 GrbasDataset 사용 (Normalize 옵션 확인 필요)
    # 학습 때 normalize_labels=True로 했다면 여기서도 해야 점수 스케일이 맞음
    ds = GrbasDataset(csv_path=CSV_PATH, normalize_labels=True)
    
    # Runners
    grad_runner = CnnGradCAM(model)
    rl_runner = GrbasCnnRLCAM(model)
    
    print(f"Starting CNN Comparison for {len(ds)} samples...")
    
    # 20개 샘플만 테스트 (너무 많으면 오래 걸림)
    for i in tqdm(range(min(len(ds), 20)), desc="Processing"):
        try:
            sample = ds[i]
            # CNN Dataset은 {feat, label, vowel_id, patient_id} 반환
            
            feat = sample['feat'].unsqueeze(0).to(DEVICE)     # (1, 1, F, T)
            vowel_id = sample['vowel_id'].unsqueeze(0).to(DEVICE) # (1,)
            pid = sample['patient_id']
            
            # 입력 Gradient 활성화 (그래프 연결용)
            feat.requires_grad = True
            
            spectrogram = feat[0, 0].detach().cpu().numpy()
            
            for item_idx, item_name in enumerate(ITEMS):
                
                # 1. Grad-CAM
                grad_mask, pred_score = grad_runner(feat, vowel_id, item_idx)
                
                # 2. RL-CAM
                rl_mask, _ = rl_runner.generate(
                    feat, vowel_id, 
                    item_idx=item_idx, 
                    steps=STEPS, 
                    lr=0.1, 
                    lambda_reg=LAMBDA_REG
                )
                
                # 저장 (예측 점수는 정규화된 0~1일 수 있으므로 5 곱해서 표기)
                # 만약 ds.y_max가 5.0이라면:
                final_score = pred_score * 5.0
                
                filename = f"{pid}_{item_name}_Score{final_score:.1f}.png"
                save_path = os.path.join(SAVE_DIR, filename)
                
                save_comparison_image(
                    spectrogram, grad_mask, rl_mask,
                    title=f"Patient {pid} | {item_name} (Pred: {final_score:.2f})",
                    save_path=save_path
                )
                
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    print(f"Done! Check '{SAVE_DIR}'")

if __name__ == "__main__":
    main()
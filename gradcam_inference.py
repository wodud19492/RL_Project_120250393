import torch
import numpy as np
from models_v3 import AutoGRBASModel, PatientGrbasDistDataset
from gradcam import GrbasGradCAM, show_cam_on_image

# 1. 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_crnn_5587.pt"  # 학습된 모델 경로
CSV_PATH = "GRBAS_test.csv" # 테스트 데이터

# 2. 모델 로드
# (주의: Patient 데이터셋이면 in_channels=3, 아니면 1)
model = AutoGRBASModel(in_channels=3).to(DEVICE)

# CRNN Lazy Init을 위한 더미 입력 (필수)
dummy = torch.zeros(1, 3, 80, 100).to(DEVICE)
model.eval()
model.encoder(dummy) 

# 가중치 로드
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# 3. Grad-CAM 설정
# 타겟 레이어: Encoder 내의 CNN의 마지막 블록의 'conv' 부분
# 구조: model -> encoder -> cnn (Sequential) -> [Block1, Block2, Block3, Dropout]
# Block3 -> net (Sequential) -> [Conv2d, BN, ReLU, MaxPool]
# 여기서 Conv2d의 출력을 보거나, ReLU 지난 후를 봐도 됨.
# 가장 일반적인 타겟: 마지막 Conv2d 레이어
target_layer = model.encoder.cnn[2].net[0] 
# (ConvBlock 정의에 따라 구조가 다를 수 있으니 print(model)로 확인 추천)

grad_cam = GrbasGradCAM(model, target_layer)

# 4. 데이터 하나 가져오기
ds = PatientGrbasDistDataset(csv_path=CSV_PATH)
sample = ds[2] # 첫 번째 환자 데이터
input_tensor = sample['feat'].unsqueeze(0).to(DEVICE) # (1, 3, 80, T)
input_tensor.requires_grad = True # 입력 자체 gradient는 필요 없지만 안전하게

# 5. 실행 (예: 'G' 항목에 대해)
ITEMS = ["G", "R", "B", "A", "S"]
item_idx = 0 # G (Grade)

print(f"Generating Grad-CAM for Item: {ITEMS[item_idx]}...")

# Grad-CAM 계산
# class_idx=None이면 모델이 예측한 클래스(예: G=2)를 설명
mask, predicted_class = grad_cam(input_tensor, item_idx=item_idx, class_idx=None)

print(f"Predicted Class Score: {predicted_class + 1}") # 0~4 idx -> 1~5 score

# 6. 시각화
# Patient Dataset은 채널이 3개(/a/, /i/, /u/)입니다.
# 시각화할 때는 그 중 하나(예: /a/)를 골라서 보여주거나 평균을 냅니다.
# 여기선 /a/ (Channel 0) 위에 Heatmap을 겹쳐보겠습니다.
spectrogram = input_tensor[0, 0, :, :].cpu().detach().numpy() # (80, T)

title = f"Grad-CAM for {ITEMS[item_idx]} (Pred: {predicted_class+1})"
show_cam_on_image(spectrogram, mask, title=title, save_path="grad_cam.png")
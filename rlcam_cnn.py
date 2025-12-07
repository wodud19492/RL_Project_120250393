import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLCamGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # [전략 변경] "조각하기(Sculpting)" 초기화
        # Weight=0, Bias=2.0 으로 설정하면 Sigmoid(2.0) ≈ 0.88
        # 즉, 처음에 마스크가 거의 1(Red)인 상태로 시작해서
        # 점수에 기여하지 않는 부분을 "깎아내리는(0으로 만드는)" 방식으로 학습 유도
        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 2.0) 

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class GrbasCnnRLCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_map = None
        
        # 마지막 ConvBlock의 첫 번째 Conv2d 찾기
        last_block = self.model.features[-1]
        target_layer = last_block.net[0]
        
        target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.feature_map = output

    def generate(self, feat, vowel_id, item_idx=0, steps=100, lr=0.1, lambda_reg=0.1):
        """
        Hyperparameter Tuning:
        - steps: 100회
        - lr: 0.1
        - lambda_reg: 0.1 (너무 작으면 안 깎이고, 너무 크면 다 깎임. 0.1 추천)
        """
        device = feat.device
        self.model.eval()
        
        # 1. 원본 예측 (Target Score 기준점 잡기)
        with torch.no_grad():
            org_scores = self.model(feat, vowel_id)
            org_features = self.feature_map.detach()
            # 우리가 '보존'해야 할 원래 점수
            original_score_val = org_scores[0, item_idx].item()
            
        # 2. Generator 초기화
        in_channels = org_features.shape[1]
        generator = RLCamGenerator(in_channels).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        
        # [핵심 수정 1] 패딩 감지 로직 (dB 스케일 대응)
        # 단순히 0이 아니라, 해당 스펙트로그램의 '최솟값(Min)' 근처를 배경으로 간주
        min_val = feat.min()
        # 최솟값보다 조금이라도 큰 부분만 '신호'로 간주 (Thresholding)
        # 예: -100dB가 최저라면 -95dB 이상인 부분만 마스크 허용
        is_not_padding = (feat > (min_val + 5.0)).float() 

        # 3. Optimization Loop
        for i in range(steps):
            optimizer.zero_grad()
            
            mask_small = generator(org_features)
            
            # Upsampling
            mask_upsampled = F.interpolate(
                mask_small, 
                size=feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # 패딩 부분은 강제로 0으로 깎아버림 (Gradient도 흐르지 않게 detach 안 함)
            mask_upsampled = mask_upsampled * is_not_padding
            
            masked_feat = feat * mask_upsampled
            
            # 예측
            pred_scores = self.model(masked_feat, vowel_id)
            current_score = pred_scores[0, item_idx]
            
            # [핵심 수정 2] "점수 보존(Preserve)" Loss
            # 목표: 마스크를 씌워도 원래 점수(original_score_val)와 똑같이 나와야 함
            # 차이(Diff)를 줄이는 것이 목표
            score_loss = torch.abs(current_score - original_score_val)
            
            # Regularization (Sparsity)
            # 마스크 크기(L1 norm)를 줄여라 -> 불필요한 부분은 꺼라
            reg_loss = torch.mean(mask_small)
            
            # 최종 Loss
            # 점수 유지가 규제보다 훨씬 중요하므로 가중치 조절
            # (점수 차이가 0.1만 나도 큼, 마스크 평균은 0~1 사이)
            loss = score_loss + lambda_reg * reg_loss
            
            loss.backward()
            optimizer.step()

        final_mask = mask_upsampled.detach().cpu().numpy()[0, 0]
        
        # Normalize (시각화를 위해 0~1로 펼쳐줌)
        if final_mask.max() - final_mask.min() > 1e-8:
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
            
        return final_mask, pred_scores[0, item_idx].item()
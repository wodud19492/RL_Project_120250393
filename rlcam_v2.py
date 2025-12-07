import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLCamGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # [Sculpting 초기화]
        # Bias=2.0 -> Sigmoid(2.0) ≈ 0.88 (붉은색으로 시작)
        # 불필요한 부분을 깎아내는 방식
        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 2.0)

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class GrbasRLCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_map = None
        
        # 타겟 레이어: Encoder 내의 CNN의 마지막 ConvBlock의 첫 번째 Conv2d
        # AutoGRBASModel 구조에 따라 인덱싱
        target_layer = self.model.encoder.cnn[2].net[0]
        target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.feature_map = output

    def generate(self, input_tensor, item_idx=0, steps=100, lr=0.5, lambda_reg=0.05):
        """
        [CRNN 최적화 튜닝]
        - lr: 0.5 (RNN Gradient Vanishing 극복을 위해 매우 높게 설정)
        - lambda_reg: 0.05 (약한 Gradient가 Reg에 먹히지 않도록 규제 완화)
        """
        device = input_tensor.device
        
        # [모드 설정] RNN Backward 허용
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
                module.eval()
        
        # 1. 원본 점수 계산 (Target Score)
        with torch.no_grad():
            _, org_scores = self.model(input_tensor, return_scores=True)
            org_features = self.feature_map.detach()
            original_score_val = org_scores[0, item_idx].item()

        # 2. Generator 초기화
        in_channels = org_features.shape[1]
        generator = RLCamGenerator(in_channels).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        
        # [핵심 수정 1] 패딩 감지 로직 (Max 기준 Dynamic Range)
        # 소리의 최댓값(Max)을 기준으로 60dB 아래는 잡음/침묵으로 간주
        max_val = input_tensor.max()
        threshold = max_val - 60.0  # Dynamic Range 60dB
        is_not_padding = (input_tensor > threshold).float()

        # 3. Optimization Loop
        for i in range(steps):
            optimizer.zero_grad()
            
            mask_small = generator(org_features)
            
            # Upsampling
            mask_upsampled = F.interpolate(
                mask_small, 
                size=input_tensor.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # 패딩 마스킹 (배경 지우기)
            mask_upsampled = mask_upsampled * is_not_padding
            
            # 마스크 적용
            masked_input = input_tensor * mask_upsampled
            
            # 예측
            _, pred_scores = self.model(masked_input, return_scores=True)
            current_score = pred_scores[0, item_idx]
            
            # [핵심 수정 2] Score Preservation Loss
            score_loss = torch.abs(current_score - original_score_val)
            
            # Regularization (Sparsity)
            reg_loss = torch.mean(mask_small)
            
            # 최종 Loss
            loss = score_loss + lambda_reg * reg_loss
            
            loss.backward()
            optimizer.step()

        final_mask = mask_upsampled.detach().cpu().numpy()[0, 0]
        
        # Normalize (시각화를 위해 0~1)
        if final_mask.max() - final_mask.min() > 1e-8:
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
            
        return final_mask, pred_scores[0, item_idx].item()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLCamGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        # [초기화] Bias=0.0 -> Sigmoid(0) = 0.5 (회색)
        # 이제 교체 마스킹을 쓸 것이므로, 중간값에서 시작해 
        # 중요한 건 1로, 안 중요한 건 0으로 이동하게 합니다.
        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class GrbasRLCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_map = None
        
        # Target Layer: CNN Encoder Last Conv
        target_layer = self.model.encoder.cnn[2].net[0]
        target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.feature_map = output

    def generate(self, input_tensor, item_idx=0, steps=100, lr=0.1, lambda_reg=0.1):
        """
        [Final Fix]
        - Masking: Replacement Strategy (Background Substitution)
        """
        device = input_tensor.device
        
        # Mode: Hybrid
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
                module.eval()
        
        # 1. Target Score & Background Value
        with torch.no_grad():
            _, org_scores = self.model(input_tensor, return_scores=True)
            org_features = self.feature_map.detach()
            target_score_val = org_scores[0, item_idx].item()
            
            # [중요] 배경값(Silence) 추출
            # 전체 텐서에서 가장 작은 값을 '침묵'으로 정의
            bg_val = input_tensor.min()

        # 2. Generator
        in_channels = org_features.shape[1]
        generator = RLCamGenerator(in_channels).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

        # 3. Optimization Loop
        for i in range(steps):
            optimizer.zero_grad()
            
            mask_small = generator(org_features)
            
            mask_upsampled = F.interpolate(
                mask_small, 
                size=input_tensor.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # [핵심 수정] Replacement Masking
            # 마스크가 1이면 원본(input), 0이면 배경값(bg_val)
            # 수식: M*I + (1-M)*BG
            masked_input = (input_tensor * mask_upsampled) + \
                           (bg_val * (1 - mask_upsampled))
            
            # 예측
            _, pred_scores = self.model(masked_input, return_scores=True)
            current_score = pred_scores[0, item_idx]
            
            # Loss: (Original과 점수 차이 최소화) + (마스크 최소화)
            # MSE Loss가 미분이 부드러워서 수렴이 잘 됨
            score_loss = (current_score - target_score_val) ** 2
            reg_loss = torch.mean(mask_small)
            
            loss = score_loss + lambda_reg * reg_loss
            
            loss.backward()
            optimizer.step()

        final_mask = mask_upsampled.detach().cpu().numpy()[0, 0]
        
        # Normalize
        if final_mask.max() - final_mask.min() > 1e-8:
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
            
        return final_mask, pred_scores[0, item_idx].item()
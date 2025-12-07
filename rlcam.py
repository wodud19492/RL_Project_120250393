import torch
import torch.nn as nn
import torch.nn.functional as F

class RLCamGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.conv(x)
        mask = self.sigmoid(mask)
        return mask

class GrbasRLCAM:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.model.eval()
        self.feature_map = None
        self.target_layer_name = target_layer_name
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.feature_map = output

        if self.target_layer_name is not None:
            module = self.model
            for attr in self.target_layer_name.split("."):
                module = getattr(module, attr)
            target_layer = module
        else:
            target_layer = self.model.encoder.cnn[2].net[0]

        target_layer.register_forward_hook(hook_fn)

    def generate(self, input_tensor, item_idx=0, target_class=None, 
                 steps=50, lr=0.1, lambda_reg=0.1):

        device = input_tensor.device
        self.model.train()

        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.Dropout)):
                module.eval()

        with torch.no_grad():
            logits = self.model(input_tensor) 
            org_features = self.feature_map.detach()

        if target_class is None:
            target_logits = logits[0, item_idx]
            target_class = torch.argmax(target_logits).item()

        in_channels = org_features.shape[1]
        generator = RLCamGenerator(in_channels).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

        with torch.no_grad():
            min_val = input_tensor.min()
            is_not_padding = (torch.abs(input_tensor - min_val) > 1e-3).float()

        for i in range(steps):
            self.model.zero_grad()
            optimizer.zero_grad()

            mask_small = generator(org_features)

            mask_upsampled = F.interpolate(
                mask_small, 
                size=input_tensor.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

            mask_upsampled = mask_upsampled * is_not_padding
            masked_input = input_tensor * mask_upsampled

            new_logits = self.model(masked_input)

            probs = F.softmax(new_logits[0, item_idx], dim=0)
            score = probs[target_class]

            loss = score - lambda_reg * torch.mean(mask_small)
            
            loss.backward()
            optimizer.step()

        final_mask = mask_upsampled.detach().cpu().numpy()[0, 0]

        if final_mask.max() - final_mask.min() > 1e-8:
            final_mask = (final_mask - final_mask.min()) / (final_mask.max() - final_mask.min())
            
        return final_mask, target_class

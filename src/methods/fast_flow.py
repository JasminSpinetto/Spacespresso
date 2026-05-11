from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from typing import Any

from src.methods.base import BaseMethod


class FastFlowBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x, rev=False):
        # Split channels for affine coupling
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not rev:
            # Forward transformation towards the normal distribution
            out = self.conv(x1)
            s, t = torch.chunk(out, 2, dim=1)
            x2 = x2 * torch.exp(s) + t
            return torch.cat([x1, x2], dim=1), s.sum(dim=(1, 2, 3))
        else:
            # Inverse transformation (for inference/generation)
            out = self.conv(x1)
            s, t = torch.chunk(out, 2, dim=1)
            x2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, x2], dim=1)

# --- Main Method Class ---

class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        method_cfg = config.get("method", {})
        # FastFlow performs better with lightweight architectures like ResNet18
        self.backbone_name = method_cfg.get("backbone", "resnet18") 
        self.flow_steps = method_cfg.get("flow_steps", 8)
        self.lr = method_cfg.get("lr", 1e-3)
        self.epochs = method_cfg.get("epochs", 15)
        self.image_size = config.get("data", {}).get("image_size", 224)
        self.sigma = method_cfg.get("sigma", 4)

        # 1. Feature Extractor (Fixed Backbone)
        self.feature_extractor = timm.create_model(
            self.backbone_name, pretrained=True, features_only=True, out_indices=(1, 2, 3)
        ).to(self.device)
        self.feature_extractor.eval()

        # 2. Build Flows for each output scale of the backbone
        dummy_in = torch.zeros(1, 3, self.image_size, self.image_size).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(dummy_in)
        
        self.flows = nn.ModuleList([
            nn.ModuleList([FastFlowBlock(f.shape[1]) for _ in range(self.flow_steps)])
            for f in features
        ]).to(self.device)

        self.optimizer = torch.optim.Adam(self.flows.parameters(), lr=self.lr)

    def fit(self, train_data, val_data=None):
        self.flows.train()
        print(f"Training FastFlow for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(train_data, desc=f"Epoch {epoch+1}"):
                imgs = batch['image'].to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(imgs)
                
                loss = 0
                for i, feat in enumerate(features):
                    z = feat
                    log_jacob_det = 0
                    for block in self.flows[i]:
                        z, ljd = block(z)
                        log_jacob_det += ljd
                    
                    # Log-Likelihood Loss: ||z||^2 / 2 - log_jacob_det
                    loss += 0.5 * torch.sum(z**2, dim=(1, 2, 3)) - log_jacob_det
                
                loss = loss.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Loss: {total_loss/len(train_data):.4f}")
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        self.flows.eval()
        predictions = {}

        with torch.no_grad():
            for sample in tqdm(test_data, desc="FastFlow Inference"):
                img_tensor = sample.image_tensor.to(self.device).unsqueeze(0)
                features = self.feature_extractor(img_tensor)
                
                final_map = torch.zeros([self.image_size, self.image_size], device=self.device)
                
                for i, feat in enumerate(features):
                    z = feat
                    for block in self.flows[i]:
                        z, _ = block(z)
                    
                    # The anomaly map is the L2 norm of the vectors in the latent space.
                    # Anomalous pixels will not "fit" the Gaussian and will have high norms.
                    anomaly_map = torch.norm(z, p=2, dim=1).squeeze(0)
                    
                    # Upsample to original image size
                    anomaly_map = F.interpolate(
                        anomaly_map.unsqueeze(0).unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode='bilinear', align_corners=False
                    ).squeeze()
                    
                    final_map += anomaly_map

                amap_np = final_map.cpu().numpy()
                if self.sigma > 0:
                    amap_np = gaussian_filter(amap_np, sigma=self.sigma)
                
                predictions[sample.image_id] = amap_np

        return predictions
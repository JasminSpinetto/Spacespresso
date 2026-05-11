from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Any

from src.methods.base import BaseMethod



class DeBottleneck(nn.Module):
    """Simple decoder to project features back."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                       stride=stride, padding=1, output_padding=stride-1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class StudentDecoder(nn.Module):
    """Student that attempts to reconstruct the Teacher's layers."""
    def __init__(self, feature_dims: list[int]):
        super().__init__()
        # Assuming we use 3 feature levels (e.g., layer1, layer2, layer3)
        # feature_dims comes in the encoder order: [dim1, dim2, dim3]
        self.layer3 = DeBottleneck(feature_dims[2], feature_dims[1], stride=2)
        self.layer2 = DeBottleneck(feature_dims[1], feature_dims[0], stride=2)
        self.layer1 = nn.Conv2d(feature_dims[0], feature_dims[0], kernel_size=3, padding=1)

    def forward(self, x_list):
        # x_list contains [f1, f2, f3] from the teacher
        f1, f2, f3 = x_list
        
        out3 = self.layer3(f3)
        # Summation or concatenation can be used here for refinement
        out2 = self.layer2(out3 + f2) 
        out1 = self.layer1(out2 + f1)
        
        return [out1, out2, out3]

# --- Main Method Class ---

class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configuration
        method_cfg = config.get("method", {})
        self.backbone_name = method_cfg.get("backbone", "wide_resnet50_2")
        self.learning_rate = method_cfg.get("lr", 0.001)
        self.epochs = method_cfg.get("epochs", 10)
        self.batch_size = method_cfg.get("batch_size", 16)
        self.image_size = config.get("data", {}).get("image_size", 224)
        self.sigma = method_cfg.get("sigma", 4)

        # 1. Teacher (Encoder) - Frozen
        self.teacher = timm.create_model(self.backbone_name, pretrained=True, 
                                         features_only=True, out_indices=(1, 2, 3)).to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Get feature dimensions dynamically
        dummy_in = torch.zeros(1, 3, self.image_size, self.image_size).to(self.device)
        with torch.no_grad():
            dummy_out = self.teacher(dummy_in)
        feature_dims = [f.shape[1] for f in dummy_out]

        # 2. Student (Decoder)
        self.student = StudentDecoder(feature_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.learning_rate)

    def _loss_func(self, t_features, s_features):
        """Calculates the cosine distance between feature maps."""
        total_loss = 0
        for t, s in zip(t_features, s_features):
            # Normalize and calculate Mean Squared Error or Cosine Distance
            loss = 1 - F.cosine_similarity(t, s, dim=1)
            total_loss += loss.mean()
        return total_loss

    def fit(self, train_data, val_data=None):
        self.student.train()
        # Note: A real DataLoader should be used here to load train_data images
        # For brevity, I assume a conceptual loop over image tensors
        print(f"Training Reverse Distillation for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            # Implement your data loading logic here
            for batch in tqdm(train_data, desc=f"Epoch {epoch+1}"):
                imgs = batch['image'].to(self.device) # Ensure it is a tensor [B, 3, H, W]
                
                self.optimizer.zero_grad()
                with torch.no_grad():
                    t_features = self.teacher(imgs)
                
                s_features = self.student(t_features)
                loss = self._loss_func(t_features, s_features)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_data):.4f}")
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        self.student.eval()
        self.teacher.eval()
        predictions = {}

        with torch.no_grad():
            for sample in tqdm(test_data, desc="RD Inference"):
                img_tensor = sample.image_tensor.to(self.device).unsqueeze(0)
                
                t_features = self.teacher(img_tensor)
                s_features = self.student(t_features)
                
                # Generate anomaly map by combining layers
                anomaly_map = torch.ones([1, self.image_size, self.image_size], device=self.device)
                
                for t, s in zip(t_features, s_features):
                    # Pixel-wise cosine distance
                    dist = 1 - F.cosine_similarity(t, s, dim=1)
                    # Resize to original image size
                    dist = F.interpolate(dist.unsqueeze(1), size=(self.image_size, self.image_size), 
                                         mode='bilinear', align_corners=False).squeeze(1)
                    anomaly_map *= dist # Multi-scale multiplication to highlight anomalies
                
                amap_np = anomaly_map.squeeze().cpu().numpy()
                
                # Post-processing
                if self.sigma > 0:
                    amap_np = gaussian_filter(amap_np, sigma=self.sigma)
                
                # Local normalization to [0, 1] (Global _normalize_maps is applied later)
                predictions[sample.image_id] = amap_np

        return predictions
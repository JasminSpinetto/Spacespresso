## Feature Memory / Nearest-Neighbor
PaDiM — fits a multivariate Gaussian per spatial position on pretrained features, Mahalanobis distance at inference
SPADE — k-NN on pretrained features, no coreset needed, uses full training bank
CFA — coupled-hypersphere-based feature adaptation, learns a compact feature space

## Student-Teacher
Uninformed Students
STFPM (Student-Teacher Feature Pyramid Matching) — teacher-student at multiple feature pyramid levels, simpler than Uninformed Students
EfficientAD — very fast student-teacher, adds a patch description network, state-of-the-art speed/quality tradeoff
RD4AD (Reverse Distillation) — reverse the teacher-student direction, student reconstructs teacher features bottom-up
DeSTSeg — segmentation-based student-teacher with denoising

## Normalizing Flows
FastFlow — 2D normalizing flow on pretrained features, models the distribution of normal patches
CFlow-AD — conditional normalizing flow, class-conditional anomaly scoring
DifferNet — normalizing flow on image-level features

## Reconstruction-Based
Autoencoder (MSE) — stub already in repo
VAE — variational autoencoder, reconstruction + KL divergence as anomaly score
DRAEM — discriminatively trained reconstruction anomaly estimation, trains on synthetic anomalies
RealNet — reconstruction with feature-level constraints
UniFormaly — unified framework combining reconstruction and feature matching

## Diffusion-Based
DiAD — diffusion model for anomaly detection, reconstructs normal version then computes difference
AnomalyDiffusion — uses diffusion model inpainting to reconstruct anomalous regions
AnomDiff — denoising diffusion for anomaly map generation

## Self-Supervised / Synthetic Anomaly
CutPaste — self-supervised, creates synthetic anomalies by cutting and pasting image patches
DRAEM (also fits here) — uses Perlin noise + DTD textures to generate synthetic anomalies
NSA (Natural Synthetic Anomalies) — uses Poisson image editing for realistic synthetic defects
SimpleNet — simple anomaly feature adaptation using a small MLP, trained with synthetic anomalies
DevNet — deviation network, uses labeled anomaly examples as weak supervision

## Foundation Model / CLIP-Based
WinCLIP — zero/few-shot anomaly detection using CLIP with sliding window
AnomalyGPT — uses large vision-language models for anomaly reasoning
APRIL-GAN — adapts CLIP features with a lightweight GAN
InCTRL — in-context learning for anomaly detection with vision-language models

## Multi-Scale / Pyramid
Hierarchical Transformer — applies transformer at multiple scales
MSFlow — multi-scale normalizing flows
PyramidFlow — pyramid pooling + normalizing flows

## Transformer / Attention
PatchCore with ViT backbone — swap WideResNet for DINOv2/ViT-B
UniAD — unified transformer-based anomaly detection across all classes at once
DiAD (also fits here)
AnoViT — ViT-based reconstruction with attention maps as anomaly scores

## Dataset-Specific Strategies (free gains)
Multi-view fusion — max/average scores across all 5 views of the same product
Ensemble — weighted average of multiple method predictions
Using labeled anomalies — 235 labeled examples with masks for score calibration, hard negative mining, or supervised fine-tuning
Higher resolution — 320×320 or 448×448 instead of 224×224
Test-time augmentation — average predictions over flipped/rotated versions of each test image

# First to try given our T4 constraints:
Multi-view fusion (no retraining)
PatchCore with DINOv2 backbone (config change only)
Ensemble of existing methods
EfficientAD or STFPM
CutPaste / DRAEM synthetic anomaly training
WinCLIP (zero-shot baseline)
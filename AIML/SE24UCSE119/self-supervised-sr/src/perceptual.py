import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return self.l1(self.feature_extractor(pred_n), self.feature_extractor(target_n))

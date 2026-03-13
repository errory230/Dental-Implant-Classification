import torch
import torch.nn as nn
import timm

class ImplantConvNeXtTinyModel(nn.Module):
    """
    ConvNeXt(tiny)-based model
    - Backbone: ConvNeXt-Tiny (Pre-trained)
    - Output: Logits, Features
    """
    def __init__(self, num_classes):
        super(ImplantConvNeXtTinyModel, self).__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True)
        self.feature_dim = 768
        self.backbone.head.fc = nn.Identity()
        self.fc = nn.Linear(self.feature_dim, num_classes)
    def forward(self, x):
        feat = self.backbone(x)
        logits = self.fc(feat)
        return logits, feat

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_model = ImplantConvNeXtTinyModel(num_classes=10).to(device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    logits, features = sample_model(dummy_input)
    
    print(f"Logits shape: {logits.shape}")   # [1, 10] <- ten known implant fixture models
    print(f"Feature shape: {features.shape}") # [1, 768] <- 768 latent features

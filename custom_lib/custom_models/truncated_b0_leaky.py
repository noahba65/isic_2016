import torch
import torchvision.models as models
from torch import nn


class truncated_b0_leaky(nn.Module):
    def __init__(self, num_classes, removed_layers, batch_size, image_size, pretrained, dropout_p):
        super().__init__()

        # Load EfficientNet-B0 with or without pre-trained weights
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)

        # Truncate the EfficientNet backbone
        layers = 9 - removed_layers
        self.effnet_truncated = nn.Sequential(*list(effnet.features.children())[:layers])

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dynamically calculate the input size for the fully connected layer
        with torch.no_grad():
            dummy_input = torch.randn(batch_size, 3, image_size, image_size)  # Keep batch size
            dummy_output = self.effnet_truncated(dummy_input)
            dummy_output = self.global_avg_pool(dummy_output)
            fc_input_size = dummy_output.view(dummy_output.size(0), -1).size(1)  # Flatten and get size

        # Alternative classification head with LeakyReLU
        self.fc_lrelu = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p, inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.effnet_truncated(x)  # Feature extraction
        x = self.global_avg_pool(x)  # Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_lrelu(x)  # Classification
        return x



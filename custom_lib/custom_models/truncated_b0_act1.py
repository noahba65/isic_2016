import torch
import torchvision.models as models
from torch import nn
from torchvision.ops import Conv2dNormActivation

class truncated_b0_act1(nn.Module):
    def __init__(self, num_classes, removed_layers, batch_size, image_size, pretrained, dropout_p):
        super().__init__()


        # Load EfficientNet-B0 with or without pre-trained weights
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)

        # Truncate the EfficientNet backbone
        layers = 8 - removed_layers
        self.effnet_truncated = nn.Sequential(*list(effnet.features.children())[:layers])


        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dynamically calculate the input size for the fully connected layer
        with torch.no_grad():  # Disable gradient tracking for this forward pass
            dummy_input = torch.randn(batch_size, 3, image_size, image_size)  # Example input (batch_size=1, channels=3, height=224, width=224)
            dummy_output = self.effnet_truncated(dummy_input)
            act_in_channels = dummy_output.shape[1]  

        self.activation = Conv2dNormActivation(act_in_channels, 
                                               out_channels=112,
                                               kernel_size=(1, 1),
                                               stride=(1, 1),
                                               bias=False,
                                               norm_layer=nn.BatchNorm2d,
                                               activation_layer=nn.SiLU   
                                               )
        

        self.classifier = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(112, num_classes)
        )

    def forward(self, x):
        x = self.effnet_truncated(x)  # Extract features
        x = self.activation(x)
        x = self.global_avg_pool(x)  # Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
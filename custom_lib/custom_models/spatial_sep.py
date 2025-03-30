from torch import nn
import torch

class SpatialSeparableBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=6, stride=1, expansion_factor=6,
                 reduction=4, survival_prob=0.8):
        super(SpatialSeparableBlock, self).__init__()

        self.skip_connection = (stride == 1 and n_in == n_out)
        intermediate_channels = int(n_in * expansion_factor)
        padding = (kernel_size - 1) // 2
        reduced_dim = max(1, int(n_in // reduction))  # Avoid division by zero
        
        # Expand input if expansion is greater than 1
        self.expand = nn.Conv2d(n_in, intermediate_channels, kernel_size=1, bias=False) if expansion_factor > 1 else nn.Identity()

        # Spatially separable convolution (depthwise)
        self.conv1k = nn.Conv2d(intermediate_channels, intermediate_channels, (1, kernel_size), 
                                stride=(1, stride), padding=(0, padding), groups=intermediate_channels, bias=False)

        self.convk1 = nn.Conv2d(intermediate_channels, intermediate_channels, (kernel_size, 1), 
                                stride=(stride, 1), padding=(padding, 0), groups=intermediate_channels, bias=False)

        # Squeeze and Excitation Block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(intermediate_channels, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, intermediate_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Pointwise convolution to match output channels
        self.pointwise_conv = nn.Conv2d(intermediate_channels, n_out, kernel_size=1, bias=False)
        
        # Batch Norm and Activation
        self.batch_norm = nn.BatchNorm2d(intermediate_channels)
        self.activation = nn.SiLU() 

        # Stochastic Depth (Dropout)
        self.drop_layers = nn.Dropout2d(1 - survival_prob) if survival_prob < 1.0 else nn.Identity()

    def forward(self, x):
        residual = x  # Save input for skip connection

        x = self.expand(x)
        x = self.conv1k(x)
        x = self.convk1(x)
        
        x = self.batch_norm(x)


        x = self.activation(x)


        x = self.se(x)  # Apply Squeeze and Excitation
        x = self.pointwise_conv(x)

        if self.skip_connection:
            x = self.drop_layers(x)  # Apply stochastic depth
            x += residual  # Skip connection

        return x

class SpatialSeparableCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(SpatialSeparableCNN, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.layers = nn.Sequential(
            SpatialSeparableBlock(32, 64, kernel_size=5, stride=1),
            SpatialSeparableBlock(64, 128, kernel_size=3, stride=2),
            SpatialSeparableBlock(128, 256, kernel_size=3, stride=2),
            SpatialSeparableBlock(256, 512, kernel_size=3, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Pooling
        self.fc = nn.Linear(512, num_classes)  # Fully Connected Layer

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
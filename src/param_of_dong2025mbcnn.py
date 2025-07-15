"""Count the number of parameters of Dong et al."""

import torch
import torch.nn as nn

from param_of_khan_model import count_trainable_params

class PreprocessingBlock(nn.Module):
    """Preprocessing component with parallel 1x1 and 3x3 convolutions"""
    
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3):
        super(PreprocessingBlock, self).__init__()
        
        # 1x1 convolution branch
        self.conv1x1 = nn.Conv2d(in_channels, out_channels_1x1, kernel_size=1, padding=0)
        self.bn1x1 = nn.BatchNorm2d(out_channels_1x1)
        
        # 3x3 convolution branch
        self.conv3x3 = nn.Conv2d(in_channels, out_channels_3x3, kernel_size=3, padding=1)
        self.bn3x3 = nn.BatchNorm2d(out_channels_3x3)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # 1x1 branch
        x1 = self.conv1x1(x)
        x1 = self.bn1x1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        
        # 3x3 branch
        x2 = self.conv3x3(x)
        x2 = self.bn3x3(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        
        # Concatenate along channel dimension
        out = torch.cat([x1, x2], dim=1)
        return out


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block with residual connections"""
    
    def __init__(self, in_channels, out_channels, scaling_factor=0.1):
        super(MultiScaleBlock, self).__init__()
        self.scaling_factor = scaling_factor
        
        # Calculate branch channels to ensure proper concatenation
        branch1_channels = out_channels // 3
        branch2_channels = out_channels // 3
        branch3_channels = out_channels - branch1_channels - branch2_channels  # Handle remainder
        
        # Multiple 1x1 convolutions (parallel branches)
        self.conv1x1_1 = nn.Conv2d(in_channels, branch1_channels, kernel_size=1)
        self.bn1x1_1 = nn.BatchNorm2d(branch1_channels)
        
        self.conv1x1_2 = nn.Conv2d(in_channels, branch2_channels, kernel_size=1)
        self.bn1x1_2 = nn.BatchNorm2d(branch2_channels)
        
        self.conv1x1_3 = nn.Conv2d(in_channels, branch3_channels, kernel_size=1)
        self.bn1x1_3 = nn.BatchNorm2d(branch3_channels)
        
        # 3x3 convolutions following 1x1 convolutions
        self.conv3x3_1 = nn.Conv2d(branch1_channels, branch1_channels, kernel_size=3, padding=1)
        self.bn3x3_1 = nn.BatchNorm2d(branch1_channels)
        
        self.conv3x3_2 = nn.Conv2d(branch2_channels, branch2_channels, kernel_size=3, padding=1)
        self.bn3x3_2 = nn.BatchNorm2d(branch2_channels)
        
        # Additional processing layers
        self.conv_final = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn_final = nn.BatchNorm2d(out_channels)
        
        # Residual connection adjustment
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # Get branch channel sizes
        total_channels = self.conv_final.in_channels
        branch1_channels = total_channels // 3
        branch2_channels = total_channels // 3
        branch3_channels = total_channels - branch1_channels - branch2_channels
        
        # Branch 1: 1x1 -> BN -> ReLU -> 3x3 -> BN -> ReLU
        x1 = self.conv1x1_1(x)
        x1 = self.bn1x1_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3x3_1(x1)
        x1 = self.bn3x3_1(x1)
        x1 = self.relu(x1)
        
        # Branch 2: 1x1 -> BN -> ReLU -> 3x3 -> BN -> ReLU
        x2 = self.conv1x1_2(x)
        x2 = self.bn1x1_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv3x3_2(x2)
        x2 = self.bn3x3_2(x2)
        x2 = self.relu(x2)
        
        # Branch 3: 1x1 -> BN -> ReLU
        x3 = self.conv1x1_3(x)
        x3 = self.bn1x1_3(x3)
        x3 = self.relu(x3)
        
        # Concatenate branches
        out = torch.cat([x1, x2, x3], dim=1)
        
        # Final convolution
        out = self.conv_final(out)
        out = self.bn_final(out)
        
        # Apply scaling factor to the processed features
        out = out * self.scaling_factor
        
        # Adjust residual connection if needed
        if residual.shape[1] != out.shape[1]:
            residual = self.residual_conv(residual)
        
        # Add residual connection
        out = out + residual
        out = self.relu(out)
        
        return out


class MBCNN(nn.Module):
    """Multi-Branch Convolutional Neural Network (MBCNN)"""
    
    def __init__(self, input_channels=1, num_classes=2):
        super(MBCNN, self).__init__()
        
        # Preprocessing component
        # Input size: nx1x256x256 -> nx384x63x63
        # Need to design blocks to achieve 384 channels and 63x63 spatial size
        # 256 -> 128 -> 64 -> 63 (approximately 4x reduction)
        self.pre_block1 = PreprocessingBlock(input_channels, 96, 96)  # -> nx192x128x128
        self.pre_block2 = PreprocessingBlock(192, 96, 96)  # -> nx192x64x64
        
        # Final preprocessing to reach exactly 384 channels and 63x63
        self.pre_final = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((63, 63))
        )
        
        # Multi-scale feature extraction
        # Transform from nx384x63x63 to nx2144x15x15
        # Need to reduce spatial size from 63x63 to 15x15 (approximately 4x reduction)
        self.ms_block1 = MultiScaleBlock(384, 512)
        self.ms_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 63x63 -> 31x31
        
        self.ms_block2 = MultiScaleBlock(512, 768)
        self.ms_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 31x31 -> 15x15
        
        self.ms_block3 = MultiScaleBlock(768, 1024)
        self.ms_block4 = MultiScaleBlock(1024, 1536)
        self.ms_block5 = MultiScaleBlock(1536, 2144)
        
        # Ensure exactly 15x15 size
        self.feature_pool = nn.AdaptiveAvgPool2d((15, 15))
        
        # Classifier component
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(2144, num_classes)
    
    def forward(self, x):
        # Preprocessing: nx1x256x256 -> nx384x63x63
        x = self.pre_block1(x)
        x = self.pre_block2(x)
        x = self.pre_final(x)  # Ensure exactly 384 channels and 63x63 size
        
        # Multi-scale feature extraction: nx384x63x63 -> nx2144x15x15
        x = self.ms_block1(x)
        x = self.ms_pool1(x)  # Reduce spatial size
        
        x = self.ms_block2(x)
        x = self.ms_pool2(x)  # Reduce spatial size
        
        x = self.ms_block3(x)
        x = self.ms_block4(x)
        x = self.ms_block5(x)
        
        # Ensure exactly 15x15 size
        x = self.feature_pool(x)
        
        # Classifier: nx2144x15x15 -> nx2
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    model = MBCNN(input_channels=1, num_classes=2)
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 256, 256)
    
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {count_trainable_params(model)}")
    
    model.eval()
    with torch.no_grad():
        x = input_tensor
        print(f"\nIntermediate shapes:")
        print(f"Input: {x.shape}")
        
        # Preprocessing
        x = model.pre_block1(x)
        print(f"After pre_block1: {x.shape}")
        x = model.pre_block2(x)
        print(f"After pre_block2: {x.shape}")
        x = model.pre_final(x)
        print(f"After pre_final: {x.shape}\n")
        
        # Multi-scale feature extraction
        x = model.ms_block1(x)
        print(f"After ms_block1: {x.shape}")
        x = model.ms_pool1(x)
        print(f"After ms_pool1: {x.shape}")
        
        x = model.ms_block2(x)
        print(f"After ms_block2: {x.shape}")
        x = model.ms_pool2(x)
        print(f"After ms_pool2: {x.shape}")
        
        x = model.ms_block3(x)
        print(f"After ms_block3: {x.shape}")
        x = model.ms_block4(x)
        print(f"After ms_block4: {x.shape}")
        x = model.ms_block5(x)
        print(f"After ms_block5: {x.shape}")
        x = model.feature_pool(x)
        print(f"After feature_pool: {x.shape}\n")
        
        # Classifier
        x = model.avgpool(x)
        print(f"After avgpool: {x.shape}")
        x = model.flatten(x)
        print(f"After flatten: {x.shape}")
        x = model.classifier(x)
        print(f"Final output: {x.shape}")

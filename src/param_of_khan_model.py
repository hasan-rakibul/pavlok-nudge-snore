"""Just to count the number of parameters of Khan et al."""

import torch.nn as nn

class Khan2DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Conv2D_1
            nn.ReLU(),  # Activation_1
            nn.Conv2d(32, 32, kernel_size=3),  # Conv2D_2
            nn.ReLU(),  # Activation_2
            nn.MaxPool2d(kernel_size=2),  # MaxPooling2D_1
            nn.Dropout(0.25),  # Dropout_1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv2D_3
            nn.ReLU(),  # Activation_3
            nn.Conv2d(64, 64, kernel_size=3),  # Conv2D_4
            nn.ReLU(),  # Activation_4
            nn.MaxPool2d(kernel_size=2),  # MaxPooling2D_2
            nn.Dropout(0.25)  # Dropout_2
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten
            nn.Linear(64 * 6 * 6, 512),  # Dense_1
            nn.ReLU(),  # Activation_5
            nn.Dropout(0.5),  # Dropout_3
            nn.Linear(512, 64),  # Dense_2
            nn.ReLU(),  # Activation_6
            nn.Linear(64, 1),  # Dense_3
            nn.Sigmoid()  # Activation_7
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Function to calculate trainable parameters
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize the model
model = Khan2DCNN()

# Calculate the number of trainable parameters
num_trainable_params = count_trainable_params(model)

print(f"Total trainable parameters: {num_trainable_params}\n")

# also by torchsummary
# import torchsummary
# print(torchsummary.summary(model, (1, 32, 32)))

import torch
import lightning as L

import torch.nn as nn
import torch.nn.functional as F

class CNN1D(L.LightningModule):
    def __init__(self, input_shape, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[0], 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (input_shape[1] // 4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

        self.criterion = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(1) # remove the last dimension to match the label shape
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
        return optimizer
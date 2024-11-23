import torch
import lightning as L
from torchmetrics.classification import BinaryAccuracy
import torch.nn as nn
import torch.nn.functional as F

from param_of_khan_model import Khan2DCNN

class CNN1D(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_conv = len(config.arch.out_channels)
        self.pooling_kernel_size = config.arch.pooling_kernel_size

        self.conv_layers = nn.ModuleList()
        in_channels = config.data.n_mfcc
        for i in range(self.num_conv):
            out_channels = config.arch.out_channels[i]
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1
                )
            )
            in_channels = out_channels
        
        self.fc = nn.Linear(
            in_features=in_channels * (config.data.max_mfcc_length // (self.num_conv * self.pooling_kernel_size)),
            out_features=1
        )

        self.dropout = nn.Dropout(config.arch.dropout)

        self.criterion = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()
        
    def forward(self, x):
        for i in range(self.num_conv):
            x = F.relu(self.conv_layers[i](x))
            x = F.max_pool1d(x, self.pooling_kernel_size)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x.squeeze(1) # remove the last dimension to match the label shape
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, sync_dist=True)
        pred_labels = y_pred.sigmoid().round().int()
        truth_labels = y.int()
        acc = BinaryAccuracy().to(self.device)(pred_labels, truth_labels)
        self.log('val_acc', acc, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
        return optimizer

class Khan2DCNNForNumericMFCC(Khan2DCNN):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten
            nn.Linear(4864, 512),  # Dense_1 ## changed from 64 * 6 * 6 to 4864 to match the input shape
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

class Khan2DCNNLightning(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = Khan2DCNNForNumericMFCC()
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension for Conv2D
        return self.model(x).squeeze(1)  # remove the last dimension to match the label shape
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, sync_dist=True)
        pred_labels = y_pred.sigmoid().round().int()
        truth_labels = y.int()
        acc = BinaryAccuracy().to(self.device)(pred_labels, truth_labels)
        self.log('val_acc', acc, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
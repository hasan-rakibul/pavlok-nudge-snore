import torch
import lightning as L
import torchmetrics as tm
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
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        # Calculate classification metrics
        y = y.to("cpu") # move to cpu for metrics calculation
        y_pred = torch.round(torch.sigmoid(y_pred)).to("cpu")
        y = y.int()
        y_pred = y_pred.int()
        print(f"DEBUG: {y} is predicted as {y_pred}")
        accuracy = tm.classification.BinaryAccuracy()(y_pred, y)
        precision = tm.classification.BinaryPrecision()(y_pred, y)
        recall = tm.classification.BinaryRecall()(y_pred, y)
        f1_score = tm.classification.BinaryF1Score()(y_pred, y)
        auc = tm.classification.BinaryAUROC()(y_pred, y)

        self.log('test_accuracy', accuracy, sync_dist=True)
        self.log('test_precision', precision, sync_dist=True)
        self.log('test_recall', recall, sync_dist=True)
        self.log('test_f1_score', f1_score, sync_dist=True)
        self.log('test_auc', auc, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
        return optimizer
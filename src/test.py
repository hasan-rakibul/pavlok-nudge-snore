import os
from omegaconf import OmegaConf
import torch
import lightning as L
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocessing import get_test_dataloader
from model import CNN1D 

def main():
    config_file = "./config/config.yaml"
    config = OmegaConf.load(config_file)

    L.seed_everything(config.seed)

    assert config.checkpoint, "Checkpoint is required for testing the model without training"
    config.logging_dir = os.path.join(*config.checkpoint.split("/")[:3])

    print(f"\nLoading checkpoint from {config.checkpoint}")
    model = CNN1D.load_from_checkpoint(config.checkpoint)

    test_loader = get_test_dataloader(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    model = model.to(device)
    model.eval()
    y_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_pred = model(x.to(device))
            y_pred = torch.round(torch.sigmoid(y_pred))
            y_all.extend(y.cpu().int().tolist())
            y_pred_all.extend(y_pred.cpu().int().tolist())

    metrics = {
        "Accuracy": accuracy_score(y_all, y_pred_all),
        "Precision": precision_score(y_all, y_pred_all),
        "Recall": recall_score(y_all, y_pred_all),
        "F1 Score": f1_score(y_all, y_pred_all),
        "AUC": roc_auc_score(y_all, y_pred_all)
    }

    with open(os.path.join(config.logging_dir, "test_metrics.txt"), "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
            print(f"{metric}: {value}")       

if __name__ == "__main__":
    main()
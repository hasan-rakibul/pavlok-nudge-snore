import os
import datetime
from omegaconf import OmegaConf
import lightning as L

from preprocessing import get_train_val_dataloader, get_test_dataloader
from model import CNN1D

def main():
    config_file = "./config/config.yaml"
    config = OmegaConf.load(config_file)

    L.seed_everything(config.seed)

    if config.checkpoint:
        # use >./logs/... as logging_dir if checkpoint is provided
        print(f"\nLoading checkpoint from {config.checkpoint}")
        logging_dir = os.path.join(*config.checkpoint.split("/")[:3])
    else:
        logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )
        os.makedirs(logging_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(logging_dir, "config.yaml"))

    config.logging_dir = logging_dir # update logging_dir to use later

    if config.do_train:
        train_loader, val_loader = get_train_val_dataloader(config)

    if config.do_test:
        test_loader = get_test_dataloader(config)

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        deterministic=True,
        devices="auto",
        accelerator="auto",
        log_every_n_steps=3,
        default_root_dir=config.logging_dir,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filename="best_model",
                save_top_k=1,
                mode="min"
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min"
            )
        ]
    )

    if config.checkpoint:
        model = CNN1D.load_from_checkpoint(config.checkpoint)
    else:
        model = CNN1D(
            input_shape=(config.data.n_mfcc, config.data.max_mfcc_length),
            config=config
        )

    if config.do_train:
        print("\nTraining the model...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        model = CNN1D.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    if config.do_test:
        print("\nTesting the model...")
        import torch
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = model.to(device)
        y_all, y_pred_all = [], []
        for batch in test_loader:
            x, y = batch
            y_pred = model(x.to(device))
            y_pred = torch.round(torch.sigmoid(y_pred))
            y_all.extend(y.int().tolist())
            y_pred_all.extend(y_pred.int().tolist())
    
        accuracy = accuracy_score(y_all, y_pred_all)
        precision = precision_score(y_all, y_pred_all)
        recall = recall_score(y_all, y_pred_all)
        f1 = f1_score(y_all, y_pred_all)
        auc = roc_auc_score(y_all, y_pred_all)
        with open(os.path.join(logging_dir, "test_metrics.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"AUC: {auc}\n")

if __name__ == "__main__":
    main()
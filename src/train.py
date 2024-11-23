import os
import datetime
from omegaconf import OmegaConf
import lightning as L
import argparse

from preprocessing import get_train_val_dataloader
from model import CNN1D, Khan2DCNNLightning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--khan", action="store_true", help="Use the Khan et al. model")

    args = parser.parse_args()
    if args.khan:
        print("Using the Khan et al. model")
        config_file = "./config/config_khan.yaml"
    else:
        print("Using the proposed model")
        config_file = "./config/config.yaml"

    config = OmegaConf.load(config_file)

    L.seed_everything(config.seed)

    if config.train_checkpoint:
        # use >./logs/... as logging_dir if checkpoint is provided
        print(f"\nLoading checkpoint from {config.train_checkpoint}")
        logging_dir = os.path.join(*config.train_checkpoint.split("/")[:3])
    else:
        logging_dir=os.path.join(
            config.logging_dir, 
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + config.expt_name
        )

    config.logging_dir = logging_dir # update logging_dir to use later

    train_loader, val_loader = get_train_val_dataloader(config)
    print("Number of training samples:", len(train_loader.dataset))
    print("Number of validation samples:", len(val_loader.dataset), "\n")
        
    if not args.khan:
        callbacks = [
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss",
                save_top_k=1,
                mode="min"
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min"
            )
        ]
    else:
        # Khan did not mention any early stopping in their paper
        # so, just saving the last epoch
        callbacks = [
            L.pytorch.callbacks.ModelCheckpoint(
                save_top_k=1
            )
        ]

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        deterministic=True,
        devices="auto",
        accelerator="auto",
        log_every_n_steps=3,
        default_root_dir=config.logging_dir,
        callbacks=callbacks
    )

    if config.train_checkpoint:
        model = CNN1D.load_from_checkpoint(config.train_checkpoint) if not args.khan \
            else Khan2DCNNLightning.load_from_checkpoint(config.train_checkpoint)
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=config.train_checkpoint)
    else:
        model = CNN1D(config) if not args.khan else Khan2DCNNLightning(config.train.lr)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.validate(model, dataloaders=val_loader, verbose=True)      

if __name__ == "__main__":
    main()
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
        devices = "auto" # use all GPUs if available

    if config.do_test:
        test_loader = get_test_dataloader(config)
        devices = 1 # recommended to use 1 GPU for testing

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        deterministic=True,
        devices=devices,
        accelerator="auto",
        log_every_n_steps=3,
        default_root_dir=config.logging_dir,
        # callbacks=[
        #     ModelCheckpoint(
        #         monitor="val_loss",
        #         filename="best_model",
        #         save_top_k=1,
        #         mode="min"
        #     )
        # ]
    )

    if config.checkpoint:
        model = CNN1D.load_from_checkpoint(config.checkpoint)
    else:
        model = CNN1D(
            input_shape=(config.data.n_mfcc, config.data.max_mfcc_length),
            config=config
        )

    if config.do_train:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        model = CNN1D.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    if config.do_test:
        metrics = trainer.test(model, dataloaders=test_loader)
        with open(os.path.join(logging_dir, "test_metrics.txt"), "w") as f:
            f.write(str(metrics))

if __name__ == "__main__":
    main()
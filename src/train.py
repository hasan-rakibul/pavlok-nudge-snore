import os
from omegaconf import OmegaConf
import lightning as L

from preprocessing import LitDataModule
from model import CNN1D

def main():
    config_file = "./config/config.yaml"
    config = OmegaConf.load(config_file)

    L.seed_everything(config.seed)

    lit_data_module = LitDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        n_mfcc=config.data.n_mfcc,
        max_mfcc_length=config.data.max_mfcc_length,
        num_workers=config.data.num_workers
    )

    model = CNN1D(
        input_shape=(config.data.n_mfcc, lit_data_module.max_mfcc_length),
        config=config
    )

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        deterministic=True,
        devices="auto",
        accelerator="auto",
        log_every_n_steps=10
    )

    trainer.fit(model, lit_data_module)

if __name__ == "__main__":
    main()
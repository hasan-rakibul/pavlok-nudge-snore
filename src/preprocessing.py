import os
import torch
import torchaudio
import lightning as L
import numpy as np
import random

class DataModule(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_mfcc, max_mfcc_length):
        self.data_dir = data_dir
        self.n_mfcc = n_mfcc

        self.audio_files = []
        self.labels = []
        for sub_dir in os.listdir(data_dir):
            sub_dir_path = os.path.join(data_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                for file in os.listdir(sub_dir_path):
                    self.audio_files.append(os.path.join(sub_dir_path, file))
                    self.labels.append(int(sub_dir)) # labels are the subdirectorie names (0 for non-snore and 1 for snore)

        self.max_mfcc_length = max_mfcc_length

    # def _calculate_max_mfcc_length(self):
    #     max_length = 0
    #     for audio_file in self.audio_files:
    #         mfcc = self._calculate_mfcc(audio_file)
    #         if mfcc.size(2) > max_length:
    #             max_length = mfcc.size(2)
    #     return max_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.int8)

        mfcc = self._calculate_mfcc(audio_file)
        
        return mfcc, label
    
    def _calculate_mfcc(self, audio_file):
        waveform, sr = torchaudio.load(audio_file, normalize=True) # returns ([channel, time], sample_rate)

        # if the audio is stereo, convert it to mono
        if waveform.size(0) == 2: 
            waveform = waveform.mean(dim=0, keepdims=True)

        transform = torchaudio.transforms.MFCC(
            sample_rate = sr,
            n_mfcc = self.n_mfcc,
            melkwargs={"n_fft": 2048, "hop_length": 512, "center": False} # selected default values from https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
        )

        mfcc = transform(waveform) # returns ([channel, n_mfcc, time]); torch.Size([1, 25, 83]) for 1 sec audio
        mfcc = torch.nn.functional.pad(mfcc, (0, self.max_mfcc_length - mfcc.size(2)), mode="reflect") # pad the mfcc to the max length
        
        mfcc = mfcc.squeeze(0) # remove the channel dimension
        
        return mfcc
    
# taken from https://pytorch.org/docs/stable/notes/randomness.html
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) 

def _get_dataloader(dataset, config, shuffle):
    # making sure the shuffling is reproducible
    g = torch.Generator()
    g.manual_seed(config.seed)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        worker_init_fn=_seed_worker,
        generator=g
    )

def get_train_val_dataloader(config):
    assert 'train' in os.listdir(config.data.data_dir) and 'val' in os.listdir(config.data.data_dir), 'Train and val directory found.'    
    train_dir = os.path.join(config.data.data_dir, 'train')
    val_dir = os.path.join(config.data.data_dir, 'val')
    train_dataset = DataModule(train_dir, config.data.n_mfcc, config.data.max_mfcc_length)
    val_dataset = DataModule(val_dir, config.data.n_mfcc, config.data.max_mfcc_length)

    train_loader = _get_dataloader(train_dataset, config, shuffle=True)
    val_loader = _get_dataloader(val_dataset, config, shuffle=False)

    return train_loader, val_loader
    
def get_test_dataloader(config):
    assert 'test' in os.listdir(config.data.data_dir), 'Test directory not found.'    
    test_dir = os.path.join(config.data.data_dir, 'test')
    test_dataset = DataModule(test_dir, config.data.n_mfcc, config.data.max_mfcc_length)
    test_loader = _get_dataloader(test_dataset, config, shuffle=False)
    return test_loader


class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, n_mfcc, max_mfcc_length, num_workers):
        super().__init__()

        assert 'train' in os.listdir(data_dir) and 'val' in os.listdir(data_dir) and 'test' in os.listdir(data_dir), \
             'Data directory must contain train, val and test directories.'
        
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')
        self.test_dir = os.path.join(data_dir, 'test')
        assert os.path.exists(self.train_dir) and os.path.exists(self.val_dir) and os.path.exists(self.test_dir), "Data directories not found, check the path."
        
        self.batch_size = batch_size
        self.n_mfcc = n_mfcc
        self.num_workers = num_workers

        self.max_mfcc_length = max_mfcc_length
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = DataModule(data_dir=self.train_dir, n_mfcc=self.n_mfcc, max_mfcc_length=self.max_mfcc_length)
            self.val_data = DataModule(self.val_dir, self.n_mfcc, self.max_mfcc_length)
        if stage == 'test' or stage is None:
            self.test_data = DataModule(self.test_dir, self.n_mfcc, self.max_mfcc_length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
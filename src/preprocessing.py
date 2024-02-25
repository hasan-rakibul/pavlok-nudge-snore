import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torchaudio
import lightning as L

class DataModule(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_mfcc=25):
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

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]

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

        return mfcc, label

class LitDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, n_mfcc=25):
        super().__init__()
        
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')
        self.test_dir = os.path.join(data_dir, 'test')
        assert os.path.exists(self.train_dir) and os.path.exists(self.val_dir) and os.path.exists(self.test_dir), "Data directories not found, check the path."
        
        self.batch_size = batch_size
        self.n_mfcc = n_mfcc

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = DataModule(self.train_dir, self.n_mfcc)
            self.val_data = DataModule(self.val_dir, self.n_mfcc)
        if stage == 'test' or stage is None:
            self.test_data = DataModule(self.test_dir, self.n_mfcc)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
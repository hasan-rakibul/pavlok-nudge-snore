# snore-detection

## Environment
This repository is developed and tested on the following environment:
- OS: SUSE Linux Enterprise Server 15 SP4
- Python 3.12.3, package are listed in `requirements.txt` and `requirements_toch_rocm.txt`, specific versions of major packages are:
    - torch==2.4.1+rocm6.1
    - torchaudio==2.4.1+rocm6.1
    - torchmetrics==1.4.1
    - pytorch-lightning==2.4.0
- ffmpeg 4.4.1, required for audio processing, installed in the OS.
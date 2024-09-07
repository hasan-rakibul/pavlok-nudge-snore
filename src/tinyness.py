import torch
from omegaconf import OmegaConf

from fvcore.nn import FlopCountAnalysis
from calflops import calculate_flops

from model import CNN1D

def count_flops():
    config_file = "./config/config.yaml"
    config = OmegaConf.load(config_file)

    model = CNN1D(config)
    sample_input = torch.randn(1, config.data.n_mfcc, config.data.max_mfcc_length)
    flops = FlopCountAnalysis(model, sample_input)
    print(flops.total()/1e3, "K FLOPs") # 67.464 K FLOPs

    flops, macs, params = calculate_flops(model= model, input_shape = (1, config.data.n_mfcc, config.data.max_mfcc_length),
                                          output_as_string=True)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params)) # FLOPs:138.98 KFLOPS   MACs:67.46 KMACs   Params:1.34 K

    # Note: they don't match and so I am yet to find a reliable library to calculate FLOPs

if __name__ == "__main__":
    count_flops()
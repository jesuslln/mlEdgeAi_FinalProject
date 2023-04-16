import argparse
import numpy as np
from onnxruntime.quantization.quantize import quantize_static
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization import CalibrationDataReader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from set_seed import set_random_seed


parser = argparse.ArgumentParser(description='EE361K project get latency')
parser.add_argument('--onnx_model', required=True, type=str,
                    help='Path of FLOAT ONNX model [FLOAT ONLY!]')
args = parser.parse_args()

set_random_seed(233)

input_onnx = args.onnx_model
infer_path = input_onnx.replace('.onnx', '_infer.onnx')
output_path = input_onnx.replace('.onnx', '_int8.onnx')

tds = CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
]), download=True)

vds = CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
]))
tdl = DataLoader(tds, batch_size=256, shuffle=True, num_workers=2)
vdl = DataLoader(vds, batch_size=256, shuffle=False, num_workers=2)


class CIFAR10Reader(CalibrationDataReader):
    def __init__(self, ds, num) -> None:
        super().__init__()
        self.ds = ds
        self.num = num
        self.cnt = 0

    def get_next(self) -> dict:
        if self.cnt >= self.num:
            return None
        img = self.ds[self.cnt][0].unsqueeze(0).numpy().astype(np.float32)
        self.cnt += 1
        if self.cnt % 100 == 0:
            print(self.cnt)
        return {'input': img}


try:
    quant_pre_process(input_onnx, infer_path)
    cifar10_dr = CIFAR10Reader(tds, 1000)
    quantize_static(infer_path, output_path, cifar10_dr)
except:
    cifar10_dr = CIFAR10Reader(tds, 1000)
    quantize_static(input_onnx, output_path, cifar10_dr)

from autoencoder import AutoEncoder
import torch

import torch.nn as nn
class DepthEncoder(nn.Module) :
    def __init__(self):
        super().__init__()
        self.autoencoder = AutoEncoder((3, 128, 128), 32)
        self.autoencoder.load_state_dict(torch.load('autoencoder.pth',map_location=torch.device('cpu')))
    def forward(self,x):
        x = x.float() / 255.0
        return self.autoencoder(x)

model =DepthEncoder()
model.eval()


dummy_input = torch.randn(1, 3, 128, 128)  # batch=1
torch.onnx.export(
    model,                   # your PyTorch model
    dummy_input,               # dummy input for tracing
    "autoencoder.onnx",            # output file
    export_params=True,        # store trained weights
    opset_version=11,          # ONNX opset
    input_names=["input"],     # input tensor name
    output_names=["output"],   # output tensor name
    dynamic_axes={             # optional: allow variable batch size
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)
class Encoder(nn.Module) :
    def __init__(self):
        super().__init__()
        self.autoencoder = AutoEncoder((3, 128, 128), 32)
        self.autoencoder.load_state_dict(torch.load('autoencoder.pth',map_location=torch.device('cpu')))
    def forward(self,x):
        x = x.float() /255
        return self.autoencoder.encode(x)

encoder = Encoder()
encoder.eval()
dummy_input = torch.randn(1, 3, 128, 128)  # batch=1
torch.onnx.export(
    encoder,                   # your PyTorch model
    dummy_input,               # dummy input for tracing
    "encoder.onnx",            # output file
    export_params=True,        # store trained weights
    opset_version=11,          # ONNX opset
    input_names=["input"],     # input tensor name
    output_names=["output"],   # output tensor name
    dynamic_axes={             # optional: allow variable batch size
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

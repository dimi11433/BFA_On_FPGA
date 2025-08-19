import torch 
import torch.nn as nn
import torchvision.models as models 
import onnx

model = models.resnet18(weights=None)
model.load_state_dict(torch.load("resnet18_weights_noquant.pth"))
model.eval()

example_inputs = (torch.randn(1, 3, 224, 224),)
torch.onnx.export(
    model,
    example_inputs,
    "resnet-18.onnx",
    export_params=True,
    opset_version=11,              # 11 or 13 are safest for hls4ml
    do_constant_folding=True,      # fold constants for simpler graph
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
# onnx_program.save("resnet-18.onnx")
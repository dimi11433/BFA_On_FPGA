import torch.onnx
import onnx
import hls4ml
# Load into hls4ml

onnx_model = onnx.load("resnet-18.onnx")

config = hls4ml.utils.config_from_onnx_model(onnx_model, granularity='model', backend='Vivado')

hls_model = hls4ml.converters.convert_from_onnx_model(
    onnx_model,
    hls_config=config,
    backend='Vivado',
    output_dir='hls4ml_prj1/',
    part='xc7z020clg400-1'
)
hls_model.compile()
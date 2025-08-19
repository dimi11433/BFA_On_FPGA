import torch 
import hls4ml 
import onnx
import torchvision.models as models 

#load the model 
model = models.resnet18(weights=None)
model.load_state_dict(torch.load("resnet18_weights_noquant.pth"))
model.eval()

# input_shape = (3, 224, 224)
# example_input = torch.randn(1, 3, 28, 28)
example_input = torch.randn(1, 3, 224, 224)

config = hls4ml.utils.config_from_pytorch_model(
    model,
    granularity='model',
    backend='Vivado',
)

hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    input_shape = (1, 3, 224, 224),
    hls_config = config,
    backend = 'Vivado', #Fix later
    output_dir = 'BFA_no_Quant/hls4ml_prj/',
    part = 'xc7z020clg400-1' #Input part numver for board
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()
# hls_model.build(csim=False)



# #Lets try loading and using an onnx module 
# model_onnx = onnx.load('resnet-18.onnx')

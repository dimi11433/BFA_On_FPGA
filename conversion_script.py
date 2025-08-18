import torch 
import hls4ml 
import torchvision.models as models 

#Load the model
model = torch.jit.load('resnet-18.pt')
model.eval()

input_shape = (3, 224, 224)

config = hls4ml.utils.config_from_pytorch_model(
    model, 
    input_shape, 
    granularity = 'model'
)
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    hls_config = config,
    backend = 'VivadoHLS', #Fix later
    output_dir = 'BFA_no_Quant/hls4ml_prj/',
    part = 'xc7z020clg400-1' #Input part numver for board
    
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()
hls_model.build(csim=False)
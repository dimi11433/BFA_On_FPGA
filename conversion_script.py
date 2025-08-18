import torch 
import hls4ml 


#Load the model
model = torch.load('resnet-18.pt')
model.eval()

input_shape = (3, 224, 224)

config = hls4ml.utils.config.config_from_pytorch_model(model, input_shape, granularity = 'model')
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    hls_config = config,
    backend = 'VivadoHLS', #Fix later
    output_dir = 'BFA_no_Quant',
    part = '' #Input part numver for board
    
    
    
)
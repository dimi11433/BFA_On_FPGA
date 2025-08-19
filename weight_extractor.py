import torch 
import torchvision.models as models

save_path = "resnet18_weights_noquant.pth"
#Load the saved model 
model = torch.jit.load('resnet-18.pt')

#The weights and model definiton is saved as dict values so we save them 
scripted_state_dict = model.state_dict()

#We then create an instnce of a cean resnet-18 model 
normal_resnet18 = models.resnet18(pretrained=False)

#We then load the weights and model definiton into the clean isntance of the resnet-18 
normal_resnet18.load_state_dict(scripted_state_dict)

normal_resnet18.eval()
torch.save(normal_resnet18.state_dict(), save_path)
print("we were successful- file saved to {save_path}")
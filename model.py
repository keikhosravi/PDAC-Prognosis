from torchvision import datasets, transforms, utils, models
import torch
import torch.nn as nn
from collections import OrderedDict



class survresnet(nn.Module):
    def __init__(self):
        super(survresnet, self).__init__()
        label_dim = 1
        use_pretrained=True 
        feature_extract = False

        PATH="/deepdata/adib/prognostic_study/pytorch_code/trained_model/trainedResnet.pth"
        model_ft = models.resnet101(pretrained=use_pretrained)       
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        input_size = 224
        model_dict = model_ft.state_dict()
        pretrained_dict = torch.load(PATH)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
        model_dict.update(pretrained_dict)
        model_ft.load_state_dict(model_dict)
    
        self.resnet = model_ft
        
    def forward(self, x):
        
        return self.resnet(x)
    
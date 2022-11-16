import torch
import torch.nn as nn
from torchvision import models

def load_model():

    folder = r'vgg19_20220918_211657'
    sub = r'vgg19_20220918_211657_fold_1'
    model_name = folder.split('_')[0]
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        root = r'/media/kv/Documents/git/mtkvcs-saved-models/'
        
        path = root + folder + r'/' + sub + r'/'
        filename = sub + r'.pt'
        path = path+filename
   
        model.classifier = nn.Sequential(nn.Linear(25088,3136), # Configure the classifier
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(3136,3136, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(3136,2, bias=True))
        
        model.load_state_dict(torch.load(path))
    
    return model, folder, root
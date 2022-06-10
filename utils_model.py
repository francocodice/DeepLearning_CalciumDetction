
import torch
import torch.nn as nn

from model import *

def load_densenet(path_model):
    model = HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(1024, 2)

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model



def test_calcium_det(path_model):
    model = HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]

    del model.fc1
    del model.fc2

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))
        
    model.load_state_dict(dict_model)

    return model


def load_densenet_mlp(path_model):
    model = HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def load_resnet_mlp(path_model):
    model = HierarchicalResidual(encoder='resnet18')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def load_effcientNet(path_model):
    model = HierarchicalResidual(encoder='efficientnet-b0')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    for param in model.parameters():
        param.requires_grad = False

    del model.fc1
    del model.fc2

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1280, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def load_densenet_mse(path_model):
    model = HierarchicalResidual(encoder='densenet121')
    dict_model = torch.load(path_model)["model"]
    model.load_state_dict(dict_model)

    del model.fc1
    del model.fc2

    for param in model.parameters():
        param.requires_grad = False

    model.fc =  torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


## unfreeze last layer backbone

def activate_denselayer16(model):
    model_last_layer = model.encoder[-3][-2].denselayer16

    for param in model_last_layer.parameters():
        param.requires_grad = True

    return model_last_layer


def activate_lastlayer_eff(model):
    model_last_layer = list(model.encoder.children())[-2]
    for param in model_last_layer.parameters():
        param.requires_grad = True

    return model_last_layer


def unfreeze_param_lastlayer_res(model):
    model_last_layer = model.encoder[-2][-1]

    for param in model_last_layer.parameters():
        param.requires_grad = True

    return model_last_layer
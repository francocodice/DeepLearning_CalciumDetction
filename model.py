import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from collections import OrderedDict


class HierarchicalResidual(torch.nn.Module):
    def __init__(self, encoder='resnet18', pretrained=True):
        super().__init__()

        self.encoder_name = encoder
        self.encoder = None
        self.num_ft = 0

        if 'resnet' in encoder:
            resnets = {
                'resnet18': torchvision.models.resnet18,
                'resnet34': torchvision.models.resnet34,
                'resnet50': torchvision.models.resnet50,
                'resnet101': torchvision.models.resnet101,
            }

            self.encoder = resnets[encoder](pretrained=pretrained)
            self.encoder.conv1.weight.data = self.encoder.conv1.weight.data[:, :1]
            self.encoder.conv1.in_channels = 1
            self.num_ft = self.encoder.fc.in_features
            self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        
        elif 'densenet' in encoder:
            self.encoder = torch.hub.load('pytorch/vision:v0.6.0', encoder, pretrained=pretrained)
            self.encoder.features.conv0.weight.data = self.encoder.features.conv0.weight.data[:, :1]
            self.encoder.features.conv0.in_channels = 1
            self.num_ft = self.encoder.classifier.in_features
            self.encoder = torch.nn.Sequential(
                self.encoder.features,
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )

        elif 'efficientnet' in encoder:
            self.encoder = EfficientNet.from_pretrained(encoder)
            self.encoder._conv_stem.weight.data = self.encoder._conv_stem.weight.data[:, :1]
            self.encoder._conv_stem.in_channels = 1
            #self.encoder = EfficientNet.from_pretrained(encoder)
            self.num_ft = self.encoder._fc.in_features

        elif 'resnext' in encoder: 
            self.encoder = torch.hub.load('pytorch/vision:v0.8.0', encoder, pretrained=True)
            self.encoder.conv1.weight.data = self.encoder.conv1.weight.data[:, :1]
            self.encoder.conv1.in_channels = 1
            self.num_ft = self.encoder.fc.in_features
            self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        else:
            print(f'Unkown encoder {encoder}')
            exit(1)

        """parent classes [
            No Finding, Enlarged Cardiomediastinum, Lung Opacity, 
            Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support devices
        ]
        """
        self.fc1 = torch.nn.Linear(in_features=self.num_ft, out_features=8, bias=True)

        """child classes [
            Cardiomegaly, Lung Lesion, Edema, Consolidation, Pneumonia, Atelactasis
        ]
        """
        self.fc2 = torch.nn.Linear(in_features=self.num_ft+8, out_features=6, bias=True)
        
        # Sort output with correct label order
        output_order = torch.tensor([2, 4, 5, 6, 7, 8, 0, 1, 3, 9, 10, 11, 12, 13])
        self.out_idx = torch.argsort(output_order)
    
    def forward(self, x):
        if 'efficientnet' in self.encoder_name:
            x = self.encoder.extract_features(x)
            x = self.encoder._avg_pooling(x)
        else:
            x = self.encoder(x)

        # correction from original code here
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


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



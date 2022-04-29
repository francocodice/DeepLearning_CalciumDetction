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



class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder5 = UNet._block(features * 8, features * 16, name="enc5")
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 16, features * 32, name="bottleneck")

        self.upconv5 = nn.ConvTranspose2d(
            features * 32, features * 16, kernel_size=2, stride=2
        )
        self.decoder5 = UNet._block((features * 16) * 2, features * 16, name="dec5")
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

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



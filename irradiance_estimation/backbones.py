import torchvision
from torch import nn

class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights=config.model.backbone_pretrained_weights)
        
        if config.model.backbone_pretrained_weights == "DEFAULT":
            for _, mod_ in self.named_modules():
                if isinstance(mod_, nn.BatchNorm2d):
                    for param in mod_.parameters():
                        param.requires_grad = False
                
        self.resnet.fc = nn.Identity()

    def _get_final_layer_shape(self):
        return 2048
        
    def forward(self, x):
        x = self.resnet(x)
        return x

class ConvNextSmall(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convnext = torchvision.models.convnext_small(weights=config.model.backbone_pretrained_weights)
        for name, param in self.convnext.named_parameters():
            if 'bn' in name:
                param.requires_grad = False

        self.convnext.classifier[-1] = nn.Identity()

    def _get_final_layer_shape(self):
        return 768
        
    def forward(self, x):
        x = self.convnext(x)
        return x
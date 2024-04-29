import torchvision
import torch.nn as nn

from timm.models.registry import register_model


class VGG16(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        net = torchvision.models.vgg16_bn(pretrained=False)
        self.features = net.features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out

@register_model
def vgg16( num_classes=10, **kwargs):
    return VGG16(num_classes)


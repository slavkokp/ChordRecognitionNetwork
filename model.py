from torch import nn
import torch.nn.functional as F
import torch
#from torchvision.models.mobilenetv2 import mobilenet_v2
#from torchvision.models.efficientnet import efficientnet_v2_s
from torchvision.models.resnet import resnet18
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_model = resnet18(pretrained=True)
        self.fc3 = [None for _ in range(6)]

        for i in range(6):
            self.fc3[i] = nn.Linear(1536, 19, device=device)

    def forward(self, x):
        result = [None for _ in range(6)]
        # for other nets like mobilenet_v2
        # x = self.features_model.features(x)
        x = self.features_model.conv1(x)
        x = self.features_model.bn1(x)
        x = self.features_model.relu(x)
        x = self.features_model.maxpool(x)

        x = self.features_model.layer1(x)
        x = self.features_model.layer2(x)
        x = torch.flatten(x, 1)
        # branching here
        for i in range(6):
            result[i] = self.fc3[i](x)

        return result
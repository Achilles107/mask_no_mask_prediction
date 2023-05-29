import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class MaskNoMaskClassifier(nn.Module):
  def __init__(self):
    super(MaskNoMaskClassifier, self).__init__()

    self.sequential = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3), # (32, 148, 148)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # (32, 74, 74),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), # (64, 72, 72)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # (64, 36, 36)
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), # (128, 34, 34)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # (128, 17, 17)
        nn.Flatten(), # 36992
    )

    self.classifier = nn.Sequential(
        nn.Linear(in_features=36992, out_features=512),
        nn.ReLU(),
        # nn.Linear(in_features=1024, out_features=512),
        # nn.ReLU(),
        nn.Linear(in_features=512, out_features=1)
    )

  def forward(self, features):
    features = self.sequential(features)
    features = self.classifier(features)

    return features

def get_model():
    path = 'mask_pred_weights.pth'
    model = MaskNoMaskClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_tensor(image_bytes):
   transformation = transforms.Compose([transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                                      transforms.Resize((150, 150)),  # resize to input shape of our CNN
                                      transforms.ToTensor()  # convert PIL to Tensor
                                      ])
   image = Image.open(io.BytesIO(image_bytes))
   return transformation(image).unsqueeze(0)

print('Done')

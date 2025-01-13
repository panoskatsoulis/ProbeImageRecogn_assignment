import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

class TinyVGG(nn.Module):
  def __init__(self):
    super(TinyVGG, self).__init__()
    self.channels = 64
    self.layers = nn.Sequential(
      ## Layer 1
      #nn.Conv2d(3, 64, kernel_size=3, padding=1),
      nn.Conv2d(3, self.channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ## Layer 2
      #nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      ## Layer 3
      #nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(self.channels * 50 * 80, 512), # layers 1-2-3 output 10 (channels) 2d-tensors of 50x80
      nn.ReLU(inplace=True),
      #nn.Linear(512, 5)  # 1-bit decision, 2 numbers for top-left, 2 numbers for width and height
      nn.Linear(512, 4)  # 2 numbers for top-left, 2 numbers for width and height
    )

  def forward(self, x):
    x = self.layers(x)
    x = self.classifier(x)
    return x

model = TinyVGG()

if __name__ == "__main__":
  from dataset import flyability_train, train_data_loader
  model.cpu()
  print(model)
  print(flyability_train[0])
  print(f" input (  no model) shape {flyability_train[0][0].shape}")
  print(f"output (  no model) shape {flyability_train[0][1].shape}")
  #print(next(iter(train_data_loader)))
  output = model(flyability_train[0][0].unsqueeze(0).cpu()) # run only input data (1 image)
  #output = model(next(iter(train_data_loader))) # run only input data (1 batch - 4 images)
  print(f"output (with model) shape {output.shape}")

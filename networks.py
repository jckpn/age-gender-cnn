from torch import nn
import torch

# BasicCNN - 1 conv layer, 1 linear layer
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(1),

            nn.LazyLinear(num_classes)
        )
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For computing predictions on single images, for testing or use
        out = self.forward(single_in)
        pred = torch.argmax(out) if self.num_classes > 1 else out[0][0]
        pred = pred.cpu().detach().numpy()
        return pred
    

# LeNet 4 - 2 conv layers, 2 linear layers
class LeNet4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(1),

            nn.LazyLinear(120),
            nn.ReLU(),
            nn.Linear(120, num_classes)
        )
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For computing predictions on single images, for testing or use
        out = self.forward(single_in)
        pred = torch.argmax(out) if self.num_classes > 1 else out[0][0]
        pred = pred.cpu().detach().numpy()
        return pred


# LeNet5 - 2 conv layers, 3 linear layers
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(1),

            nn.LazyLinear(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For computing predictions on single images, for testing or use
        out = self.forward(single_in)
        pred = torch.argmax(out) if self.num_classes > 1 else out[0][0]
        pred = pred.cpu().detach().numpy()
        return pred


# AlexNet - 5 conv layers, 3 linear layers
class AlexNet(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.num_classes = num_classes

        self.layers = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained)

        # Modify final classifier layer to have correct number of outputs
        self.layers.classifier[-1] = nn.LazyLinear(num_classes)
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For computing predictions on single images, for testing or use
        out = self.forward(single_in)
        pred = torch.argmax(out) if self.num_classes > 1 else out[0][0]
        pred = pred.cpu().detach().numpy()
        return pred
    

# VGG16 - 13 conv layers, 3 linear layers
class VGG16(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.num_classes = num_classes

        self.layers = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained)

        # Modify final classifier layer to have correct number of outputs
        self.layers.classifier[-1] = nn.LazyLinear(num_classes)
    
    def forward(self, batch_in):
        # For processing batches during training
        batch_out = self.layers(batch_in)
        return batch_out
    
    def predict(self, single_in):
        # For computing predictions on single images, for testing or use
        out = self.forward(single_in)
        pred = torch.argmax(out) if self.num_classes > 1 else out[0][0]
        pred = pred.cpu().detach().numpy()
        return pred

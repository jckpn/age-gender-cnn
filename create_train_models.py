from train import train_model
from networks import *
import torch
from torch.utils.data import random_split
from face_dataset import FaceDataset, binary_gender_label, dir_label_func
from torchvision import transforms
import tests
from preprocessors import eyealign_border_80x100


ds = FaceDataset(
    dirs=['../datasets/training/imdbwiki/imdb_crop'],
    label_func=binary_gender_label,
    processor=eyealign_border_80x100,
    max_size=100,
    transform=transforms.ToTensor())

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(ds))
train_size = len(ds) - val_size
train_set, val_set = random_split(ds, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

net = train_model(
    model=LeNet4(num_outputs=2),
    train_set=train_set,
    val_set=val_set)

tests.class_accuracy(net, val_set, print_results=True)
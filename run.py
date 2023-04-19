from train import train_model
from torch.utils.data import random_split
from face_dataset import *
from label_funcs import *
from preprocessors import *
from dstransforms import *
from networks import *
import tests

ds = FastDataset(
    dir='../processed_datasets',
    label_func=binary_gender_label,
    transform=alexnet_transform)

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(ds))
train_size = len(ds) - val_size
train_set, val_set = random_split(ds, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

alexnet = train_model(
    model=AlexNet(num_classes=2, pretrained=True),
    learning_rate=0.0005,
    train_set=train_set,
    val_set=val_set)
tests.class_accuracy(alexnet, val_set, print_results=True)
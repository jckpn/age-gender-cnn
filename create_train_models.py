from train import train_model
from networks import *
import torch
from torch.utils.data import random_split
from face_dataset import FaceDataset, binary_gender_label, dir_label_func
from torchvision import transforms
import tests
from preprocessors import eyealign_border_160x200


ds = FaceDataset(
    dirs=['../datasets/training/imdbwiki/imdb_crop'],
    label_func=binary_gender_label,
    processor=eyealign_border_160x200,
    max_size=40000,
    save_dir='../datasets/training/imdb_eyealign_2',
    print_errors=True,
    transform=transforms.ToTensor())

ds = FaceDataset(
    dirs=['../datasets/training/imdbwiki/wiki_crop'],
    label_func=binary_gender_label,
    processor=eyealign_border_160x200,
    save_dir='../datasets/training/wiki_eyealign_2',
    print_errors=True,
    transform=transforms.ToTensor())

ds = FaceDataset(
    dirs=['../datasets/training/imdb_eyealign_2',
          '../datasets/training/wiki_eyealign_2'],
    label_func=binary_gender_label,
    print_errors=True,
    # max_size=100,
    transform=transforms.ToTensor())

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(ds))
train_size = len(ds) - val_size
train_set, val_set = random_split(ds, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

net1 = train_model(
    model=LeNet4(num_outputs=2),
    train_set=train_set,
    val_set=val_set)
net2 = train_model(
    model=LeNet5(num_outputs=2),
    train_set=train_set,
    val_set=val_set)
net3 = train_model(
    model=GridAttentionNet(num_outputs=2, attention_model=LeNet4, analysis_model=LeNet5),
    train_set=train_set,
    val_set=val_set)
net4 = train_model(
    model=VariableAttentionNet(num_outputs=2, attention_model=LeNet5, analysis_model=LeNet5),
    train_set=train_set,
    val_set=val_set)

tests.class_accuracy(net1, val_set, print_results=True)
tests.class_accuracy(net2, val_set, print_results=True)
tests.class_accuracy(net3, val_set, print_results=True)
tests.class_accuracy(net4, val_set, print_results=True)
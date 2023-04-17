from train import train_model
from networks import *
import torch
from torch.utils.data import random_split
from face_dataset import *
from torchvision import transforms
import tests
from preprocessors import eyealign_border_80x100
from label_funcs import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ds = FastDataset(
#     dir='../processed_datasets/',
#     label_func=age_class_label(cutoffs=[25, 999]),
#     max_size=None,
#     transform=transforms.ToTensor())

train_set = FastDataset(
    dir='../processed_datasets/imdb_eyealign_3',
    label_func=binary_gender_label,
    transform=transforms.ToTensor())
val_set = FastDataset(
    dir='../processed_datasets/wiki_eyealign_3',
    label_func=binary_gender_label,
    transform=transforms.ToTensor())

# Split dataset into training and validation sets
# val_split_ratio = 0.2
# val_size = int(val_split_ratio * len(ds))
# train_size = len(ds) - val_size
# train_set, val_set = random_split(ds, [train_size, val_size])
# print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

net = train_model(
    model=LeNet4(2).to(device),
    learning_rate=0.0005,
    train_set=train_set,
    val_set=val_set)
tests.class_accuracy(net, val_set, print_results=True)
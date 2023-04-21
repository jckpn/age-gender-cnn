from train import train_model
from torch.utils.data import random_split
from face_dataset import *
from label_funcs import *
import preprocessors.eyealign as preprocessor
from ds_transforms import *
from networks import *
import tests

ds = FastDataset(
    dir='../other/imdb_crop',
    ds_size=20000,
    processor=preprocessor.run,
    label_func=binary_gender_label,
    transform=lenet_transform_ratio)

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(ds))
train_size = len(ds) - val_size
train_set, val_set = random_split(ds, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

model = train_model(BasicCNN(2), train_set, val_set)
tests.class_accuracy(model, val_set, print_results=True)

model = train_model(LeNet(2), train_set, val_set)
tests.class_accuracy(model, val_set, print_results=True)

ds = FastDataset(
    dir='../other/imdb_crop',
    ds_size=20000,
    label_func=binary_gender_label,
    transform=alexnet_transform_ratio)

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(ds))
train_size = len(ds) - val_size
train_set, val_set = random_split(ds, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

model = train_model(AlexNet(2, True), train_set, val_set)
tests.class_accuracy(model, val_set, print_results=True)

model = train_model(AlexNet(2, False), train_set, val_set)
tests.class_accuracy(model, val_set, print_results=True)

model = train_model(VGG16(2, True), train_set, val_set)
tests.class_accuracy(model, val_set, print_results=True)

model = train_model(VGG16(2, False), train_set, val_set)
tests.class_accuracy(model, val_set, print_results=True)
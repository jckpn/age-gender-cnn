from train import train_model
from torch.utils.data import random_split, ConcatDataset
from torch.nn import CrossEntropyLoss, MSELoss
from face_dataset import *
from label_funcs import *
import preprocessor
from ds_transforms import *
from networks import *
import tests

processor = preprocessor.run
transform = alexnet_transform(224)

adience_gender_ds = FastDataset(
    'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\datasets\\training\\adience',
    binary_gender_label, transform, processor, ds_size=5000, equalise=False, classes=2, augment=False, print_errors=False)

imdb_gender_ds = FastDataset(
    'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\other\\imdb_crop',
    binary_gender_label, transform, processor, ds_size=5000, equalise=False, classes=2, augment=False, print_errors=False) #, print_errors=False)

wiki_gender_ds = FastDataset(
    'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\datasets\\training\\raw_imdbwiki_crop\\wiki_crop',
    binary_gender_label, transform, processor, ds_size=5000, equalise=False, classes=2, augment=False, print_errors=False)

utkface_gender_ds = FastDataset(
    'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\utkface_itw',
    utkface_gender_label, transform, processor, ds_size=1000, equalise=True, classes=2, augment=False, print_errors=False)

# adience_age_ds = FastDataset(
#     'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\datasets\\training\\adience',
#     binary_gender_label, transform, processor, ds_size=1000, equalise=False, classes=2, augment=False, print_errors=True)

# imdb_age_ds = FastDataset(
#     'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\other\\imdb_crop',
#     binary_gender_label, transform, processor, ds_size=1000, equalise=False, classes=100, augment=False, print_errors=True) #, print_errors=False)

# wiki_age_ds = FastDataset(
#     'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\datasets\\training\\raw_imdbwiki_crop\\wiki_crop',
#     binary_gender_label, transform, processor, ds_size=1000, equalise=False, classes=100, augment=False, print_errors=True)

# utkface_age_ds = FastDataset(
#     'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\utkface_itw',
#     utkface_gender_label, transform, processor, ds_size=1000, equalise=False, classes=100, augment=False, print_errors=True)
    

train_val_set = ConcatDataset([wiki_gender_ds, imdb_gender_ds, adience_gender_ds])

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(train_val_set))
train_size = len(train_val_set) - val_size
train_set, val_set = random_split(train_val_set, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

test_set = utkface_gender_ds

model = train_model(AlexNet(2), train_set, val_set)
tests.mae(model, test_set, print_results=True)
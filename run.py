from train import train_model
from torch.utils.data import random_split, ConcatDataset
from torch.nn import CrossEntropyLoss, MSELoss
from face_dataset import *
from label_funcs import *
import preprocessor
from ds_transforms import *
from networks import *
import tests

# Define datasets
processor = preprocessor.processor(w=50, h=50)
transform = lenet_transform(size=50)

# adience_gender_ds = SlowDataset(
#     'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\datasets\\training\\adience',
#     age_label_all, transform, processor, ds_size=50, equalise=True, classes=2, augment=False, print_errors=False)

imdb_gender_ds = SlowDataset(
    'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\other\\imdb_crop',
    age_label_all, transform, processor, ds_size=50, equalise=True, classes=100, augment=False, print_errors=False) #, print_errors=False)

wiki_gender_ds = SlowDataset(
    'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\datasets\\training\\raw_imdbwiki_crop\\wiki_crop',
    age_label_all, transform, processor, ds_size=50, equalise=True, classes=100, augment=False, print_errors=False)

utkface_gender_ds = SlowDataset(
    'C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\utkface_itw',
    age_label_all_utk, transform, processor, ds_size=50, equalise=True, classes=100, augment=False, print_errors=False)
    
train_val_set = ConcatDataset([wiki_gender_ds, imdb_gender_ds, utkface_gender_ds])

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(train_val_set))
train_size = len(train_val_set) - val_size
train_set, val_set = random_split(train_val_set, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

test_set = val_set

model = train_model(LeNet(1), train_set, val_set, loss_fn=nn.MSELoss(), model_save_dir='../models')
tests.mae(model, test_set, print_results=True)
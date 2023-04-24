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
transform = lenet_transform_100

wiki_ds = FastDataset('C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\other\\imdb_crop', age_int_label, processor,
      transform, ds_size=1000, print_errors=False)
imdb_ds = FastDataset('C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\datasets\\training\\raw_imdbwiki_crop\\wiki_crop', age_int_label, processor,
      transform, ds_size=1000, print_errors=False)
utkface_ds = FastDataset('C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\source\\utkface_itw', utkface_age_label, processor,
      transform, ds_size=1000, print_errors=False)

train_val_set = ConcatDataset([wiki_ds, imdb_ds, utkface_ds])

# Split dataset into training and validation sets
val_split_ratio = 0.2
val_size = int(val_split_ratio * len(train_val_set))
train_size = len(train_val_set) - val_size
train_set, val_set = random_split(train_val_set, [train_size, val_size])
print(f'Split dataset into {len(train_set)} training and {len(val_set)} validation examples')

# Use val set for intra-dataset accuracy tests
test_set = val_set

model = train_model(LeNet(1), train_set, val_set, loss_fn=nn.MSELoss())
tests.mae(model, test_set, print_results=True)
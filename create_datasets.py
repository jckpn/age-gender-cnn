from face_dataset import FaceDataset, binary_gender_label
from torchvision import transforms
import tests
from preprocessors import eyealign_border_160x200


ds = FaceDataset(
    dirs=['datasets/training/imdbwiki/imdb_crop'],
    label_func=binary_gender_label,
    processor=eyealign_border_160x200,
    max_size=10000,
    save_dir='datasets/training/imdb_eyealign_2',
    transform=transforms.ToTensor())

ds = FaceDataset(
    dirs=['datasets/training/imdbwiki/wiki_crop'],
    label_func=binary_gender_label,
    processor=eyealign_border_160x200,
    max_size=10000,
    save_dir='datasets/training/wiki_eyealign_2',
    transform=transforms.ToTensor())

ds = FaceDataset(
    dirs=['datasets/training/adience'],
    label_func=binary_gender_label,
    processor=eyealign_border_160x200,
    # max_size=100,
    save_dir='datasets/training/adience_eyealign_2',
    max_size=10000,
    transform=transforms.ToTensor())
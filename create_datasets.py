from face_dataset import FaceDataset, binary_gender_label
from torchvision import transforms
import tests
from preprocessors import eyealign_border_80x100

ds = FaceDataset(
    dirs=['../datasets/training/adience'],
    label_func=binary_gender_label,
    processor=eyealign_border_80x100,
    save_dir='../datasets/adience_eyealign_3',
    transform=transforms.ToTensor())
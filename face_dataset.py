import cv2 as cv
import os
from random import shuffle
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from random import randint


class FastDataset(Dataset):
    def __init__(self, dir, label_func, transform, processor=None,
                 ds_size=None, print_errors=False, min_size=30,
                 delete_bad_files=False, equalise=False, classes=None, augment=False):
        all_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                all_paths.append(os.path.join(root, f))

        # Shuffle and limit number of files if specified
        shuffle(all_paths)
        if ds_size is not None and ds_size < 1:
                ds_size = int(ds_size*len(all_paths))

        print('Reading' if processor is None else 'Reading and processing',
            f'{ds_size} files from {dir}...')

        # Load entries into dataframe in memory - faster than reading each file
        # every time we need to access it
        self.dataframe = []

        pbar = tqdm(total=ds_size, position=0, leave=False)

        # init eq requirements
        class_goal = ds_size//classes
        eq_requirements = []
        if equalise and classes is not None:
            for i in range(classes):
                requirement = {'class': i, 'count': 0, 'aug_count': 0}
                eq_requirements.append(requirement)
            print(f'Equalising to {class_goal} entries per class...')

        path_idx = 0
        cycles = 0

        while len(self.dataframe) < ds_size:
            if path_idx >= len(all_paths):
                path_idx = 0
                cycles += 1
            path = all_paths[path_idx]
            path_idx += 1
            
            filename = os.path.basename(path)

            # Get image class label from filename
            try:
                label = label_func(filename)
            except Exception:
                continue

            if label is None:
                if print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if delete_bad_files: os.remove(path)
                continue
            elif equalise and eq_requirements[label]['count'] > class_goal:
                continue # Skip if we have enough of this class

            try:
                image = cv.imread(path)
                if processor:
                    face_images, _ = processor(image)
                    image = face_images[0] # [0] is most prevalent face in image (usually only one)

                image = Image.fromarray(image) # transform expects PIL image
                if augment is True and cycles > 0:
                    # Randomly apply various augmentations to bolster dataset
                    aug_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAdjustSharpness(sharpness_factor=randint(0,10)),
                        transforms.GaussianBlur(randint(0, 10)*2+1),
                        transforms.RandomRotation(randint(0,10), fill=255),
                        transforms.RandomPerspective(distortion_scale=0.2, fill=255),
                        transforms.RandomGrayscale(p=0.3),
                        transform,
                    ])
                    image = aug_transform(image)
                    eq_requirements[label]['aug_count'] += 1
                else:
                    image = transform(image)

                entry = {'image': image, 'label': label}
                self.dataframe.append(entry)
                if equalise: eq_requirements[label]['count'] += 1
                pbar.update(1)
                    
            except Exception as e:
                if print_errors: print(f'Skipping file {filename}: {e}')
                if delete_bad_files: os.remove(path)
                continue

        print(f'\n{len(self.dataframe)} items successfully prepared')
        if print_errors:
            if equalise:
                print(f'Equalised datset to {class_goal} images per class')
            print(f'Final class counts: {eq_requirements}')
        
        print() # newline
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        entry = self.dataframe[index]
        return entry['image'], entry['label']


class SlowDataset(Dataset):
    # Loads images from disk every time they are requested
    def __init__(self, dir, label_func, transform, processor=None,
                 print_errors=False, min_size=30, ds_size=None,
                 delete_bad_files=False, equalise=False, classes=None,
                 augment=False):
        self.dir = dir
        self.label_func = label_func
        self.processor = processor
        self.transform = transform
        self.print_errors = print_errors
        self.min_size = min_size
        self.delete_bad_files = delete_bad_files
        self.equalise = equalise
        self.classes = classes
        self.ds_size = ds_size
        self.augment = augment
        self.count = 0

        self.all_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                self.all_paths.append(os.path.join(root, f))
        shuffle(self.all_paths)

        if ds_size and ds_size < len(self.all_paths):
            self.all_paths = self.all_paths[:ds_size]

        print('Found and shuffled',
            f'{len(self.all_paths)} files from {dir}...')
        
        self.eq_requirements = []
        for i in range(classes):
            requirement = {'class': i, 'count': 0}
            self.eq_requirements.append(requirement)
        self.class_goal = ds_size//classes
        
        print() # newline

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        path = self.all_paths[index % len(self.all_paths)]

        try:
            filename = os.path.basename(path)

            # Get image class label from filename
            label = self.label_func(filename)

            if label is None:
                if self.print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if self.delete_bad_files: os.remove(path)
                return self.__getitem__(index+1)
            elif self.equalise and self.eq_requirements[label]['count'] > self.class_goal:
                return self.__getitem__(index+1) # skip if we have enough of this class

            image = cv.imread(path)
            if self.processor:
                face_images, coords = self.processor(image)
                image = face_images[0]

            image = Image.fromarray(image) # transform expects PIL image
            if self.augment and index > len(self.all_paths): # Have used all data in dataset
                    # Randomly apply various augmentations to bolster dataset
                image = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAdjustSharpness(sharpness_factor=randint(0,10)),
                    transforms.GaussianBlur(randint(0, 10)*2+1),
                    transforms.RandomRotation(randint(0,10), fill=255),
                    transforms.RandomPerspective(distortion_scale=0.2, fill=255),
                    transforms.RandomGrayscale(p=0.3),
                    self.transform,
                ])(image)
            else:
                image = self.transform(image)
            
        except Exception as e:
            if self.print_errors: print(f'Skipping file {filename}: {e}')
            if self.delete_bad_files: os.remove(path)
            return self.__getitem__(index+1)
    
        return image, label
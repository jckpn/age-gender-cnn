import torch
import cv2 as cv
import numpy as np
import os
from random import shuffle
from tqdm import tqdm as progress_bar
from torch.utils.data import Dataset
from torchvision import transforms

# cite https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


def binary_gender_label(filename):
    # Infer gender class from filename
    # e.g.  'F_30_1234.jpg' -> 0
    #       'M_30_1234.jpg' -> 1
    label = filename.split('_')[0].upper()
    label = (0 if label == 'F'
            else 1 if label == 'M'
            else None)
    return label

def dir_label_func(f):
    return int(f.split(' ')[0])


class FaceDataset(Dataset):
    def __init__(self, dirs, label_func, processor=None, transform=None,
                 max_size=None, save_dir=None, print_errors=False, min_size=30):
        self.dataframe = [] # Load entries into dataframe
                            # This should be faster than reading each file
                            # every time we want to access it

        all_paths = []
        for dir in dirs:
            for fname in os.listdir(dir):
                all_paths.append(os.path.join(dir, fname))

        # Shuffle and limit number of files if specified
        if max_size is not None and max_size < len(all_paths):
            shuffle(all_paths)
            all_paths = all_paths[:max_size]

        print('Reading' if processor is None else 'Reading and processing',
              f'{len(all_paths)} files from {dirs}...')

        if save_dir and os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        for path in progress_bar(all_paths):
            filename = os.path.basename(path)

            try:
                image = cv.imread(path)
                if processor:
                    face_images, coords = processor.run(image)
                    if len(face_images) != 1:  # We want exactly 1 face per training image
                        if print_errors: print(f'Skipping {filename}:',
                                               f'{len(face_images)} faces found in image',
                                               '(1 required)')
                        continue
                    elif coords['face_width'] < min_size:
                        if print_errors: print(f'Skipping {filename}:',
                                               f'face width {coords["face_width"]} < {min_size}')
                        continue

                    image = face_images[0]
                    if save_dir is not None:
                        save_path = os.path.join(save_dir, filename)
                        cv.imwrite(save_path, image)

                if transform:
                    image = transform(image)

                # Get image class label from filename
                label = label_func(filename)

                if label is None:
                    if print_errors: print(f'Skipping {filename}:',
                                           'label function returned None')
                    continue

                entry = {'image': image, 'label': label}
                self.dataframe.append(entry)
                
            except Exception as e:
                if print_errors: print(f'Skipping file {filename}: {e}')
                continue

                # things could do better next time etc
                # datasets with multiple faces per image
                # would require multiple labels but would
                # be possible -> improve robustness?

        print(f'{len(self.dataframe)} items successfully prepared '
               f'({len(all_paths)-len(self.dataframe)} bad items ignored)\n')
        
        if save_dir is not None:
            if len(os.listdir(save_dir)) == len(self.dataframe):
                print(f'Dataset saved to {save_dir}')
            else:
                print('Error saving dataset')
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        entry = self.dataframe[index]
        return entry['image'], entry['label']
import cv2 as cv
import os
from random import shuffle
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


class FastDataset(Dataset):
    def __init__(self, dir, label_func, processor=None, transform=None,
                 ds_size=None, save_dir=None, print_errors=False, min_size=30,
                 delete_bad_files=False):
        all_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                all_paths.append(os.path.join(root, f))

        # Shuffle and limit number of files if specified
        shuffle(all_paths)
        if ds_size is not None:
            if ds_size < 1: ds_size = int(ds_size*len(all_paths))

        print('Reading' if processor is None else 'Reading and processing',
            f'{ds_size} files from {dir}...')

        if save_dir and os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        # Load entries into dataframe in memory - faster than reading each file
        # every time we need to access it
        self.dataframe = []

        pbar = tqdm(total=ds_size, position=0, leave=False)

        path_idx = 0
        while len(self.dataframe) < ds_size:
            if path_idx >= len(all_paths):
                print('Ran out of entries - reusing files to fill specified size')
                path_idx = 0
            path = all_paths[path_idx]
            path_idx += 1
            
            filename = os.path.basename(path)

            # Get image class label from filename
            label = label_func(filename)

            if label is None:
                if print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if delete_bad_files: os.remove(path)
                continue

            try:
                image = cv.imread(path)
                if processor:
                    face_images, coords = processor(image)

                    # Run checks
                    if len(face_images) != 1:  # We want exactly 1 face per training image
                        if print_errors: print(f'Skipping {filename}:',
                                            f'{len(face_images)} faces found in image',
                                            '(1 required)')
                        if delete_bad_files: os.remove(path)
                        continue
                    if min(coords[0]['face_w'], coords[0]['face_h']) < min_size:
                        if print_errors: print(f'Skipping {filename}:',
                                            f'face width {coords["face_width"]} < {min_size}')
                        if delete_bad_files: os.remove(path)
                        continue

                    image = face_images[0]
                    if save_dir is not None:
                        save_path = os.path.join(save_dir, filename)
                        cv.imwrite(save_path, image)

                if transform:
                    image = Image.fromarray(image) # transform expects PIL image
                    image = transform(image)

                entry = {'image': image, 'label': label}
                self.dataframe.append(entry)
                pbar.update(1)
                
            except Exception as e:
                if print_errors: print(f'Skipping file {filename}: {e}')
                if delete_bad_files: os.remove(path)
                continue

                # things could do better next time etc
                # datasets with multiple faces per image
                # would require multiple labels but would
                # be possible -> improve robustness?

        print(f'\n{len(self.dataframe)} items successfully prepared ' + 
            f'({len(all_paths)-len(self.dataframe)} bad items ' +
                (f'deleted)' if delete_bad_files else 'ignored)'))
        
        if save_dir is not None:
            if len(os.listdir(save_dir)) >= len(self.dataframe):
                print(f'Dataset saved to {save_dir}')
            else:
                print('Error saving dataset')
        
        print() # newline
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        entry = self.dataframe[index]
        return entry['image'], entry['label']


class SlowDataset(Dataset):
    # Loads images from disk every time they are requested
    def __init__(self, dir, label_func, processor=None, transform=None,
                 print_errors=False, min_size=30, delete_bad_files=False):
        self.dir = dir
        self.label_func = label_func
        self.processor = processor
        self.transform = transform
        self.print_errors = print_errors
        self.min_size = min_size
        self.delete_bad_files = delete_bad_files

        self.all_paths = []
        for root, _, files in os.walk(dir):
            for f in files:
                self.all_paths.append(os.path.join(root, f))
        shuffle(self.all_paths)

        print('Found and shuffled',
            f'{len(self.all_paths)} files from {dir}...')
        
        print() # newline

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            filename = os.path.basename(path)

            image = cv.imread(path)
            if self.processor:
                face_images, coords = self.processor(image)

                # Run checks
                if len(face_images) != 1:  # We want exactly 1 face per training image
                    if self.print_errors: print(f'Skipping {filename}:',
                                        f'{len(face_images)} faces found in image',
                                        '(1 required)')
                    if self.delete_bad_files: os.remove(path)
                    return self.__getitem__(index+1)
                    
                if min(coords[0]['face_w'], coords[0]['face_h']) < self.min_size:
                    if self.print_errors: print(f'Skipping {filename}:',
                                        f'face width {coords["face_width"]} < {self.min_size}')
                    if self.delete_bad_files: os.remove(path)
                    return self.__getitem__(index+1)
                    
                image = face_images[0]

            if self.transform:
                image = Image.fromarray(image) # transform expects PIL image
                image = self.transform(image)

            # Get image class label from filename
            label = self.label_func(filename)

            if label is None:
                if self.print_errors: print(f'Skipping {filename}:',
                                    'label function returned None')
                if self.delete_bad_files: os.remove(path)
                return self.__getitem__(index+1)
            
        except Exception as e:
            if self.print_errors: print(f'Skipping file {filename}: {e}')
            if self.delete_bad_files: os.remove(path)
            return self.__getitem__(index+1)
    
        return image, label
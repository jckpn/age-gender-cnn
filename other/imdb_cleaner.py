import csv
import os

csv_path = 'imdb_train_new_1024.csv'
imdb_crop_dir = '../../datasets/training/imdbwiki/imdb_crop'

with open(csv_path, 'r') as csv_file:
    length = sum(1 for _ in csv_file) # Count lines

with open(csv_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for idx, row in enumerate(reader):
        if idx == 0: continue # Skip header

        fname = row[0]
        age = row[1]
        gender = row[2]

        try:
            old_path = os.path.join(imdb_crop_dir, fname)
            new_path = os.path.join(imdb_crop_dir, f'{gender}_{age}_{idx}.jpg')
            os.replace(old_path, new_path)

            print(f"{idx}/{length}: '{old_path}' -> '{new_path}'")
        except Exception as e:
            print(e)

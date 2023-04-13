# https://stackoverflow.com/a/43286094

csv_path = ''
imdb_crop_path = ''

import pandas as pd
chunksize = 10 ** 8
for chunk in pd.read_csv(filename, chunksize=chunksize):
    process(chunk)


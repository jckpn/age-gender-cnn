import os
 
dir = 'datasets/gender/train_set'
for idx, fname in enumerate(os.listdir(dir)):
    fpath = os.path.join(dir, fname)

    # fsize = os.stat(fpath).st_size
    # if fsize < 1024:
    #    delete file

    # age_from_fname = fname.split('_')[0]

    # if age_from_fname != fname:
    #     os.rename(fpath, dir + age_from_fname + ' ' + str(idx) + '.jpg')
    #     print(f'{fname} -> {age_from_fname + " " + str(idx) + ".jpg"}')

    ext = fname.split('.')[1]

    if fname.split(' ')[0] == 'female':
        os.rename(fpath, f'{dir}/0 {idx}.{ext}')
    elif fname.split(' ')[0] == 'male':
        os.rename(fpath, f'{dir}/1 {idx}.{ext}')

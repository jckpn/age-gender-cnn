import os
import cv2
from tqdm import tqdm

dir_to_tidy = 'adience'


faceDetector = cv2.FaceDetectorYN.create(
    "C:\\Users\\jckpn\\Documents\\YEAR 3 PROJECT\\implementation\\preprocessors\\face_detection_models\\face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

deleted = 0

for fname in tqdm(os.listdir(dir)):
    fpath = os.path.join(dir, fname)

    # Check age is reasonable
    if len(fname.split('_')) < 2:
        print(f'No age in {fname} - deleting')
        os.remove(fpath)
        deleted += 1
        continue

    # Maybe rethink this cus it could still be used for gender recognition
    # age = int(fname.split('_')[1])
    # if age < 0 or age > 120:
    #     # print(f'{idx}/{len(os.listdir(dir))}: Age {age} in {fname} - deleting')
    #     os.remove(fpath)
    #     deleted += 1
    #     continue

    # Check for exactly one face present in image
    try:
        img = cv2.imread(fpath)
        faceDetector.setInputSize((int(img.shape[1]), int(img.shape[0])))
        faces = faceDetector.detect(img)[1]

        if faces is None:
            # print(f'{idx}/{len(os.listdir(dir))}: No faces detected in {fname} - deleting')
            os.remove(fpath)
            deleted += 1
            continue

        if len(faces) != 1:
            # print(f'{idx}/{len(os.listdir(dir))}: {len(faces)} faces detected in {fname} - deleting')
            os.remove(fpath)
            deleted += 1
            continue

    except:
        # print(f'{idx}/{len(os.listdir(dir))}: Error processing file {fname} - deleting')
        os.remove(fpath)
        deleted += 1
        continue

print(f'{deleted} files deleted')
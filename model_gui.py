import torch
import cv2 as cv
import numpy as np
import preprocessors.eyealign_noborder_80x120
from networks import *


def run_models(input_img):
    face_images, coords = preprocessors.eyealign_noborder_80x120.run(input_img)
    for i in range(len(face_images)):
        face_images[i] = cv.resize(face_images[i], (40, 60))
    
    # gender_preds, age_preds = [], []
    gender_preds = []
    
    for i in range(len(face_images)):
        img = face_images[i]

        input = np.transpose(img, (2, 0, 1))
        input = torch.from_numpy(input).float()
        input = input.unsqueeze(0)

        this_face_gender = gender_cnn.predict(input)
        # this_face_age = age_cnn.predict(input)

        gender_preds.append(this_face_gender)
        # age_preds.append(this_face_age)
        
    return face_images, coords, gender_preds # , age_preds


def visualize_model(input_img, coords, gender_preds): 
    input_img = cv.addWeighted(input_img, 0.75, np.zeros(input_img.shape, input_img.dtype), 0, 0)

    for idx, e in enumerate(coords):
        face_x = e['face_x'].astype(int)
        face_y = e['face_y'].astype(int)
        face_w = e['face_w'].astype(int)
        face_h = e['face_h'].astype(int)

        # darken image to make writing more visible:
        
        input_img = cv.rectangle(input_img,
                                    (face_x, face_y),
                                    (face_x + face_w, face_y + face_h),
                                    color=(255, 255, 255),
                                    thickness=1)
        
        try:
            this_gender = gender_preds[idx]
            # this_age = age_preds[idx]

            # add gender labels
            cv.putText(
                input_img,
                text=('female ' if this_gender==0 else 'male '),
                org=(face_x, face_y-5),
                fontFace=0,
                fontScale=face_w/150,
                color=(255,255,255),
                thickness=1)
        except:
            continue
    
    return input_img


def test_from_image(image_path, win_size=720):
    image = cv.imread(image_path)
    win_title = image_path.split()[-1]
    _, coords, gender_preds = run_models(image)
    
    # # resize image
    # scale = win_size/image.shape[1]
    # aspect_ratio = image.shape[1]/image.shape[0]
    # image = cv.resize(image, (int(aspect_ratio*win_size), win_size))

    for i in range(len(coords)):
        for key, _ in coords[i].items():
            coords[i][key] *= 1.0#scale

    image = visualize_model(image, coords, gender_preds)

    cv.imshow(win_title, image)
    cv.waitKey(0)


def test_from_cam(win_size=720):
    cam_input = cv.VideoCapture(0)

    counter = 0

    while cv.waitKey(1) < 0:
        _, frame = cam_input.read()

        if counter >= 20 or counter == 0: # compute predictions every 20 frames
            _, coords, gender_preds, age_preds = run_models(frame)
            counter = 1
        else:
            _, coords, _, _ = run_models(frame)

        frame = visualize_model(frame, coords, gender_preds, age_preds)
        disp_frame = cv.resize(frame, (int(frame.shape[1]/frame.shape[0]*win_size), win_size))
        cv.imshow('cam', disp_frame)
        counter += 1


if __name__ == '__main__':
    gender_model_path = 'models\LeNet5-2_13-04-02-41.pt'

    gender_cnn = LeNet5(num_outputs=2)
    gender_cnn.load_state_dict(torch.load(gender_model_path))

    test_from_image('../archive/test images/disney.jpg')
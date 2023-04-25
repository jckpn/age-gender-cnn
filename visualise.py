import torch
import cv2 as cv
import numpy as np
import preprocessor
from networks import *
from ds_transforms import *
import os
from PIL import Image
import argparse
from torchvision import transforms


def run_model(input_img, net, processor, transform):
    preds = []
    face_images, coords = processor(input_img)
    
    for img in face_images:
        img = Image.fromarray(img) # transform expects PIL image
        img = transform(img)
        pred = net.predict(img)
        preds.append(pred)
        
    return face_images, coords, preds


def visualize_model(input_img, coords, g_preds, a_preds): 
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
            this_gender = g_preds[idx]
            this_age = a_preds[idx]

            # add labels
            cv.rectangle(input_img,
                         (face_x, face_y-14),
                         (face_x+face_w, face_y),
                         (255, 255, 255),
                         -1) # fill rectangle
            cv.putText(
                input_img,
                text=('F' if this_gender==0 else 'M') + str(int(this_age)),
                org=(face_x+1, face_y-2),
                fontFace=0,
                fontScale=0.5,
                color=(0,0,0),
                thickness=1)
        except:
            continue
    
    return input_img


def visualise_image(image_path, g_net, g_processor, g_transform, a_net,
                       a_processor, a_transform, resize=720,
                       show_processed_faces=True):
    in_image = cv.imread(image_path)

    if resize:
        if in_image.shape[0] > in_image.shape[1]:
            new_width = int(in_image.shape[1]/in_image.shape[0]*resize)
            new_height = resize
            in_image = cv.resize(in_image, (new_width, new_height))
        else:
            new_width = resize
            new_height = int(in_image.shape[0]/in_image.shape[1]*resize)
            in_image = cv.resize(in_image, (new_width, new_height))

    face_images, coords, g_preds = run_model(in_image, g_net, g_processor, g_transform)
    _, _, a_preds = run_model(in_image, a_net, a_processor, a_transform)
    out_image = visualize_model(in_image, coords, g_preds, a_preds)

    if show_processed_faces:
        # paste each image from face_images into frame
        for idx, face in enumerate(face_images):
            # transform expects PIL image
            face = Image.fromarray(face)
            face = g_transform(face)
            face = face.squeeze(0).numpy()
            if len(face.shape) == 2:
                face = cv.cvtColor(face, cv.COLOR_GRAY2RGB)
            else:
                face = np.transpose(face, (1,2,0))
            face_min, face_max = face.min(), face.max()
            face = (face - face_min) / (face_max - face_min) * 255
            face = cv.resize(face, (resize//10, resize//10))
            # convert to rgb if grayscale
            out_image[idx*face.shape[0]:(idx+1)*face.shape[0], 0:face.shape[1]] = face[:,:]

    cv.imshow(image_path, out_image)
    cv.waitKey(0)


def visualise_cam(g_net, g_processor, g_transform, a_net, a_processor,
                  a_transform, resize=720, cam_id=0, update_interval=0,
                  show_processed_faces=True):
    cam_input = cv.VideoCapture(cam_id)
    _, frame = cam_input.read() 
    win_title = 'Live (Camera Input)'
    cv.imshow(win_title, frame)

    counter = 0
    while cv.waitKey(1) < 0:
        _, frame = cam_input.read()

        if resize:
            if frame.shape[0] > frame.shape[1]:
                new_width = int(frame.shape[1]/frame.shape[0]*resize)
                new_height = resize
                frame = cv.resize(frame, (new_width, new_height))
            else:
                new_width = resize
                new_height = int(frame.shape[0]/frame.shape[1]*resize)
                frame = cv.resize(frame, (new_width, new_height))

        if counter <= 0:
            face_images, coords, g_preds = run_model(frame, g_net, g_processor, g_transform)
            _, _, a_preds = run_model(frame, a_net, a_processor, a_transform)
        elif counter >= update_interval:
            counter = 0
        
        out_image = visualize_model(frame, coords, g_preds, a_preds)

        if show_processed_faces:
            # paste each image from face_images into frame
            for idx, face in enumerate(face_images):
                # transform expects PIL image
                face = Image.fromarray(face)
                face = g_transform(face)
                face = face.squeeze(0).numpy()
                if len(face.shape) == 2:
                    face = cv.cvtColor(face, cv.COLOR_GRAY2RGB)
                else:
                    face = np.transpose(face, (1,2,0))
                face_min, face_max = face.min(), face.max()
                face = (face - face_min) / (face_max - face_min) * 255
                face = cv.resize(face, (resize//10, resize//10))
                # convert to rgb if grayscale
                out_image[idx*face.shape[0]:(idx+1)*face.shape[0], 0:face.shape[1]] = face[:,:]
                
        cv.imshow(win_title, out_image)
        counter += 0


if __name__ == '__main__':
    # parse command line arguments
    # https://docs.python.org/3/tutorial/stdlib.html#command-line-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--show-processed', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    # load models
    # todo: make this accessible via cmd
    g_path = '../models\LeNet-2_2504-1625.pt'
    g_processor = preprocessor.processor(w=50, h=50)
    g_transform = lenet_transform(size=50)

    a_path = '../models\LeNet-1_2504-1633.pt'
    a_processor = preprocessor.processor(w=50, h=50)
    a_transform = lenet_transform(size=50)

    # get model classes from path e.g. 'LeNet-2_xyz.pt' -> 'LeNet(2)'
    g_net = None
    g_architecture = os.path.basename(g_path).split('_')[0].replace('-', '(') + ')'
    exec('g_net = ' + g_architecture)
    g_net.load_state_dict(torch.load(g_path, map_location=torch.device('cpu')))
    g_net.eval()

    a_net = None
    a_architecture = os.path.basename(a_path).split('_')[0].replace('-', '(') + ')'
    exec('a_net = ' + a_architecture)
    a_net.load_state_dict(torch.load(a_path, map_location=torch.device('cpu')))
    a_net.eval()

    if args.image_path:
        visualise_image(args.image_path,
                        g_net, g_processor, g_transform,
                        a_net, a_processor, a_transform,
                        show_processed_faces=args.show_processed)
    else:
        visualise_cam(g_net, g_processor, g_transform,
                      a_net, a_processor, a_transform,
                      show_processed_faces=args.show_processed)
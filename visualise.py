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


def visualise_model(input_img, coords, g_preds, a_preds, confidence_scores): 
    # darken image to make writing more visible:
    # cite this
    input_img = cv.addWeighted(input_img, 0.75, np.zeros(input_img.shape, input_img.dtype), 0, 0)

    for idx, e in enumerate(coords):
        face_x = e['face_x'].astype(int)
        face_y = e['face_y'].astype(int)
        face_w = e['face_w'].astype(int)
        face_h = e['face_h'].astype(int)

        thickness = 1 if face_w/input_img.shape[1] < 0.1 else 2
        
        input_img = cv.rectangle(input_img,
                                    (face_x, face_y),
                                    (face_x + face_w, face_y + face_h),
                                    color=(255, 255, 255),
                                    thickness=thickness)
        
        try:
            this_gender = g_preds[idx]
            this_age = a_preds[idx]

            text = ''
            if this_gender == 0:
                text = 'Female, ' if face_w > 70 else 'F'
            else:
                text = 'Male, ' if face_w > 50 else 'M'
            # if confidence_scores:
                # Ommitted due to time constraints
            text += str(int(this_age))

            cv.putText(
                input_img,
                text=text,
                org=(face_x+1, face_y-5),
                fontFace=0,
                fontScale=max(face_w/150, 0.5),
                color=(255, 255, 255),
                thickness=thickness)
        except:
            continue
    
    return input_img

def resize_to_max(image, max_size=720):
    if image.shape[0] > image.shape[1]:
        new_width = int(image.shape[1]/image.shape[0]*max_size)
        new_height = max_size
        image = cv.resize(image, (new_width, new_height))
    else:
        new_width = max_size
        new_height = int(image.shape[0]/image.shape[1]*max_size)
        image = cv.resize(image, (new_width, new_height))
    return image


def visualise_image(image_path, g_net, g_processor, g_transform, a_net,
                       a_processor, a_transform, resize=720,
                       show_processed_faces=False, confidence_scores=False):
    in_image = cv.imread(image_path)
    if resize:
        in_image = resize_to_max(in_image, resize)

    face_images, coords, g_preds = run_model(in_image, g_net, g_processor, g_transform)
    _, _, a_preds = run_model(in_image, a_net, a_processor, a_transform)
    out_image = visualise_model(in_image, coords, g_preds, a_preds, confidence_scores)

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
                  a_transform, resize=720, cam_id=0, update_interval=10,
                  show_processed_faces=False, confidence_scores=False,
                  frame_diff_threshold=0.5):
    cam_input = cv.VideoCapture(cam_id)
    _, frame = cam_input.read() 
    win_title = 'Live (Camera Input)'
    cv.imshow(win_title, frame)
    
    coords, g_preds, a_preds = [], [], []
    frame_diff = 0
    last_frame_score = 0
    counter = 0
    
    while cv.waitKey(1) < 0:
        _, frame = cam_input.read()
        if resize:
            frame = resize_to_max(frame, resize)

        if counter >= update_interval:
            this_frame_score = np.sum(frame)/frame.size
            frame_diff = abs(this_frame_score - last_frame_score)**0.5
            last_frame_score = this_frame_score
            if frame_diff < frame_diff_threshold:
                face_images, coords, g_preds = run_model(frame, g_net, g_processor, g_transform)
                _, _, a_preds = run_model(frame, a_net, a_processor, a_transform)
            counter = 0
        counter += 1
        
        out_image = visualise_model(frame, coords, g_preds, a_preds, confidence_scores)

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
                #face = cv.resize(face, (resize//10, resize//10))
                # convert to rgb if grayscale
                out_image[idx*face.shape[0]:(idx+1)*face.shape[0], 0:face.shape[1]] = face[:,:]
                
        cv.imshow(win_title, out_image)


if __name__ == '__main__':
    # parse command line arguments
    # https://docs.python.org/3/tutorial/stdlib.html#command-line-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--show-processed', action='store_true', default=False)
    parser.add_argument('--confidence-scores', action='store_true', default=False)
    args = parser.parse_args()

    # load models
    # todo: make this accessible via cmd
    g_path = './models/LeNet-2_2504-1625.pt'
    g_processor = preprocessor.processor(w=50, h=50)
    g_transform = lenet_transform(size=50)

    a_path = './models/LeNet-1_2504-1633.pt'
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
                        show_processed_faces=args.show_processed,
                        confidence_scores=args.confidence_scores)
    else:
        visualise_cam(g_net, g_processor, g_transform,
                      a_net, a_processor, a_transform,
                      show_processed_faces=args.show_processed,
                      confidence_scores=args.confidence_scores)
import cv2 as cv
import os
import numpy as np
import math


# Customisable alignment params
dest_w = 80    # Width of final image
dest_h = 100    # Height of final image
dest_eye_y = 0.475*dest_h   # Y coord of both eyes in final image
dest_eye_x = 0.35*dest_w    # X coord of left eye in final image
                            # (right eye gets calculated for symmetry)


# Init detector
detector_model_path = os.path.dirname(__file__) + "\\face_detection_models/face_detection_yunet_2022mar.onnx"
detector = cv.FaceDetectorYN.create(
    model=detector_model_path,
    config="", # Custom config - leave blank
    input_size=(500, 500), # Default image input size - set later
    score_threshold=0.99, # Confidence threshold to return a face
    nms_threshold=0.3, # ? Check what this is
    top_k=5000) # ? Check what this is


def eye_align(image, face_coords):
    # calculate transformation matrix from above parameters:
    # get translation
    dest_left_eye_x = dest_eye_x
    dest_right_eye_x = dest_w-dest_eye_x
    translate_x = dest_left_eye_x - face_coords["left_eye_x"]
    translate_y = dest_eye_y - face_coords["left_eye_y"]

    # image = cv.line(image, (face_coords["right_eye_x"], face_coords["right_eye_y"]), (face_coords["left_eye_x"], face_coords["left_eye_y"]), (255,255,255), thickness=2)
    # sohcahtoa to get angle of rotation for eye alignment
    opp = face_coords["right_eye_y"] - face_coords["left_eye_y"]
    adj = face_coords["right_eye_x"] - face_coords["left_eye_x"]
    theta = math.atan(opp/adj)

    # get scale factor
    scale = (dest_left_eye_x - dest_right_eye_x) / (face_coords["left_eye_x"] -
                                              face_coords["right_eye_x"])

    # create matrix
    align_matrix = cv.getRotationMatrix2D(center=(int(face_coords["left_eye_x"]), int(
        face_coords["left_eye_y"])), angle=math.degrees(theta), scale=scale)

    # add transformation coords to rotation matrix
    align_matrix[0][2] += translate_x
    align_matrix[1][2] += translate_y

    # apply matrix transformation to image and crop to specified width/height

    #cv.resize(image, (int(d_width*scale), int(d_height*scale)))
    image = cv.warpAffine(image, align_matrix, (dest_w, dest_h))
    return image


def equalise(input_img):
    # Histogram equalisation: equalize each channel
    # https://msameeruddin.hashnode.dev/image-equalization-contrast-enhancing-in-python
    r, g, b = cv.split(input_img)
    r_eq = cv.equalizeHist(r)
    g_eq = cv.equalizeHist(g)
    b_eq = cv.equalizeHist(b)
    img_eq = cv.merge((r_eq, g_eq, b_eq))
    return img_eq


def run(input_img):
    # Run face detector
    input_img_w, input_img_h = input_img.shape[1], input_img.shape[0]
    detector.setInputSize((input_img_w, input_img_h))
    face_data = detector.detect(input_img)[1] # [0] is confidence [check this?], [1] is coords
    if face_data is None:
        return None # Cancel if detector fails
    
    face_imgs = []
    all_face_coords = []

    for entry in face_data:
        # Get face coords
        coords = entry[:-1].astype(np.int32)
        this_coords = {'face_x': coords[0],
                       'face_y': coords[1],
                       'face_w': coords[2],
                       'face_h': coords[3],
                       'left_eye_x': coords[4],
                       'left_eye_y': coords[5],
                       'right_eye_x': coords[6],
                       'right_eye_y': coords[7]}

        all_face_coords.append(this_coords)

        this_face_img = eye_align(input_img, this_coords) # Align image with coords
        this_face_img = equalise(this_face_img) # Equalise image
        face_imgs.append(this_face_img)

    return face_imgs, all_face_coords

if __name__ == '__main__':
    # Test preprocessor
    input_img = cv.imread('../../../../archive/group.jpg')

    face_images, face_coords = run(input_img)

    for entry in face_coords:
        
        # Add face rectangle to image
        input_img = cv.rectangle(
            input_img,
            (entry['face_x'], entry['face_y']),
            (entry['face_x'] + entry['face_w'],
             entry['face_y'] + entry['face_h']),
            color=(255, 255, 255),
            thickness=2)

        # Add eye circles to image
        input_img = cv.circle(
            input_img,
            (entry['left_eye_x'], entry['left_eye_y']),
            radius=2,
            color=(255, 255, 255),
            thickness=2)
        
        input_img = cv.circle(
            input_img,
            (entry['right_eye_x'], entry['right_eye_y']),
            radius=2,
            color=(255, 255, 255),
            thickness=2)

    cv.imshow("input image", input_img)

    all_faces = np.concatenate(face_images, axis=1)

    cv.imshow("processed", all_faces)
    cv.waitKey(0)
import cv2 as cv
import os
import numpy as np
import math


# Customisable alignment params
dest_w = 224    # Width of final image
dest_h = 224    # Height of final image
dest_eye_y = 0.5*dest_h   # Y coord of both eyes in final image
dest_eye_x = 0.375*dest_w    # X coord of left eye in final image
                            # (right eye gets calculated for symmetry)


# Init detector
detector_model_path = os.path.dirname(__file__) + "/face_detection_models/face_detection_yunet_2022mar.onnx"
detector = cv.FaceDetectorYN.create(
    model=detector_model_path,
    config="", # Custom config - leave blank
    input_size=(500, 500), # Default image input size - set later
    score_threshold=0.9, # Confidence threshold to return a face
    nms_threshold=0.3, # ? Check what this is
    top_k=5000) # ? Check what this is


def eye_align(image, face_coords):
    #  CALCULATE TRANSFORMATION MATRIX FROM ABOVE PARAMETERS
    
    # 1. get translation matrix
    dest_left_eye_x = dest_eye_x
    dest_right_eye_x = dest_w-dest_eye_x
    translate_x = dest_left_eye_x - face_coords["left_eye_x"]
    translate_y = dest_eye_y - face_coords["left_eye_y"]

    # 2. calc angle of eye misalignment with sohcahtoa
    opp = face_coords["right_eye_y"] - face_coords["left_eye_y"]
    adj = face_coords["right_eye_x"] - face_coords["left_eye_x"]
    if adj == 0: return None # avoid divide by zero error
    theta = math.atan(opp/adj)

    # 3. calc scale factor
    if face_coords["left_eye_x"] == face_coords["right_eye_x"]: return None
    scale = (dest_left_eye_x - dest_right_eye_x) / (face_coords["left_eye_x"] -
                                              face_coords["right_eye_x"])

    # 4. create alignment matrix from steps 2 and 3
    align_matrix = cv.getRotationMatrix2D(center=(int(face_coords["left_eye_x"]), int(
        face_coords["left_eye_y"])), angle=math.degrees(theta), scale=scale)

    # 5. apply translation to matrix to get final matrix
    align_matrix[0][2] += translate_x
    align_matrix[1][2] += translate_y

    # 6. apply matrix transformation to image and crop to specified width/height
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

        aligned_img = eye_align(input_img, this_coords) # Align image with given coords
        
        if aligned_img is None: continue # Skip this face is alignment fails
        aligned_eq = equalise(aligned_img) # Equalise image and add entry if alignment successful
        all_face_coords.append(this_coords)
        face_imgs.append(aligned_eq)

    return face_imgs, all_face_coords

if __name__ == '__main__':
    # Test preprocessor
    input_img = cv.imread('../../../archive/group.jpg')

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
import cv2 as cv
import numpy as np
import os

detector_model_path = os.path.dirname(__file__) + "\\face_detection_models/face_detection_yunet_2022mar.onnx"
recognizer_model_path = os.path.dirname(__file__) + "\\face_detection_models/face_recognition_sface_2021dec.onnx"

detector = cv.FaceDetectorYN.create(
    model=detector_model_path,
    config="",
    input_size=(500, 500),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000)
    
recognizer = cv.FaceRecognizerSF.create(
    model=recognizer_model_path,
    config="")


def run(input_img):
    face_imgs = []
    face_coords = []    # Return coords for model gui

    # DETECTION: Detect/extract faces from image
    input_img_w = input_img.shape[1]
    input_img_h = input_img.shape[0]
    detector.setInputSize((input_img_w, input_img_h))
    faces_coords = detector.detect(input_img)[1]

    if face_coords is None:
        return [], []
    
    for face_coords in faces_coords:
        new_face_img = recognizer.alignCrop(input_img, face_coords)
        
        r, g, b = cv.split(new_face_img)
        r = cv.equalizeHist(r)
        g = cv.equalizeHist(g)
        b = cv.equalizeHist(b)
        new_face_img = cv.merge((r, g, b))

        face_imgs.append(new_face_img)
    
    return face_imgs, faces_coords


if __name__ == '__main__':
    # Test preprocessor
    input_img = cv.imread("../../archive/test images/Picture1.jpg")

    face_images, face_coords = run(input_img)

    for e in face_coords:
        e = e.astype(int)   # Convert to int type
                            # https://note.nkmk.me/en/python-numpy-dtype-astype/

        face_x = e[0]
        face_y = e[1]
        face_w = e[2]
        face_h = e[3]

        input_img = cv.rectangle(
            input_img,
            (face_x, face_y),
            (face_x + face_w, face_y + face_h),
            color=(255, 255, 255),
            thickness=2)
        
    

        # BRIGHTNESS/CONTRAST NORMALIZATION
        # Split each channel and normalize

    b, g, r = cv.split(input_img)
    b = cv.equalizeHist(b)
    g = cv.equalizeHist(g)
    r = cv.equalizeHist(r)
    input_img = cv.merge((b, g, r))

    cv.imshow("input image", input_img)

    all_faces = np.concatenate(face_images, axis=1)
    cv.imshow("processed", all_faces)
    cv.waitKey(0)
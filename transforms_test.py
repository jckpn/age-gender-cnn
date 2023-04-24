from torchvision import transforms
from random import randint
from PIL import Image
import preprocessor
import cv2 as cv


def get_concat_h(ims):
    dst = Image.new('RGB', (224*len(ims), 224))
    for i in range(len(ims)):
        dst.paste(ims[i], (224*i, 0))
    return dst

images = []

for i in range(10):
    image = cv.imread('C:/Users/jckpn/Downloads/14837124.JPG')
    image, _ = preprocessor.run(image)
    image = image[0]
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = Image.fromarray(image) # transform expects PIL image

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAdjustSharpness(sharpness_factor=randint(0,10)),
        transforms.GaussianBlur(randint(0, 10)*2+1),
        transforms.RandomRotation(randint(0,10), fill=255),
        transforms.RandomPerspective(distortion_scale=0.2, fill=255),
        transforms.RandomGrayscale(p=0.3),
    ])

    image = transform(image)
    images.append(image)

get_concat_h(images).show()
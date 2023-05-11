# 📖 Predicting Age and Gender From Facial Images With Deep Neural Networks

## What is this?

This repository contains the source code and runnable GUI for a CNN-based facial feature extractor, which predicts the gender and age of subjects from images or live video. This project was undertaken as my chosen Final Year Project (22COZ250) at Loughborough University.

For more details on the project's development, as well as an in-depth review of the current state of automatic facial feature classifiers, see the `report.pdf`.

*Children as young as three are intrinsically capable of recognising these traits by glancing at a person's face, yet encoding this ability into a mathematical model or computer program has remained a difficult challenge for many decades. However, recent advances in machine learning and the advent of convolutional neural networks have allowed researchers to replicate similar perceptive skills on computers to a remarkable degree.* (from abstract)


## ⚙️ Installation

Running this program will require **Git**, **Anaconda** and **Python 3**.

First, clone the repo in your desired installation folder:

```sh
cd PATH/TO/INSTALLATION
git clone https://github.com/jckpn/age-gender-cnn.git
cd age-gender-cnn
```

Install required packages with pip:

```sh
torch torchvision opencv-python ipykernel tqdm
```

Download these three models and place them in the `models` folder: [AlexNet-2_genderEq_83.pt](https://drive.google.com/file/d/1eeOHTckWW01P32mfIR-D0CsTnKNYwr2D/view?usp=share_link), [AlexNet-1_ageNoEq_7.pt](https://drive.google.com/file/d/1nTz7URYHH8fWc46L8GvXmi4yiJFgAry9/view?usp=share_link), [face_detection_yunet_2022mar.onnx](https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx?raw=true)

## 💻 Use

### Run with image

To test the model on a specified image, you can run:

```sh
py visualise.py --image-path path/to/image
```

You can use an image of your own, or use one of the example images from `examples`.

### Run with webcam

To test the model with live video, run visualise.py without the --image-path argument:

```sh
py visualise.py
```

This will run the model in real-time using the first detected webcam as a live input.

### Additional arguments

You can use `--show-processed` to see the pre-processed facial images alongside the model output.

## 🤖 Training your own models

To train your own models, edit `run.py` with the relevant image directories. Images must be in the filname format `gender_age_id.*` where `gender` is a binary label of either `M` or `F` and `age` is the age as an integer. `id` can be any value and is only included to avoid filename conflicts.

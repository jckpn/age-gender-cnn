# 📖 Predicting Age and Gender From Facial Images With Deep Neural Networks

## ❓ What is this?

This repository contains the source code and runnable GUI for a CNN-based facial feature extractor, which predicts the gender and age of subjects from images or live video. This project was undertaken as my chosen Final Year Project (22COZ250) at Loughborough University.

For more details on the project's development, as well as an in-depth review of the current state of automatic facial feature classifiers, see the `report.pdf`.

*Children as young as three are intrinsically capable of recognising these traits by glancing at a person's face, yet encoding this ability into a mathematical model or computer program has remained a difficult challenge for many decades. However, recent advances in machine learning and the advent of convolutional neural networks have allowed researchers to replicate similar perceptive skills on computers to a remarkable degree.*


## ⚙️ Installation

Running this program will require **Git**, **Anaconda** and **Python 3**.

First, clone the repo in your desired installation folder:

```sh
cd PATH/TO/INSTALLATION
git clone https://github.com/jckpn/age-gender-cnn.git
cd age-gender-cnn
```

Clone the conda environment and install relevant packages:

```sh
conda env create -f environment.yml
```

Download the OpenCV face detection model from [here](https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/face_detection_yunet_2022mar.onnx?raw=true) and place it in the `age-gender-cnn` installation folder.


## 💻 Use
To test the models on an image, you can run:

```sh
python3 visualise.py --image-path path/to/image
```

You can use an image of your own, or use one of the example images from `examples`.

To use the models with live video, use:

```sh
python3 visualise.py --camera
```

This will run the model in real-time using the first detected webcam as a live input.


## 🤖 Training your own models

To train your own models, edit `run.py` with the relevant image directories. Images must be in the filname format `gender_age_id.*` where `gender` is a binary label of either `M` or `F` and `age` is the age as an integer. `id` can be any value and is only included to avoid filename conflicts.

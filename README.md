<h1 align="center"><b>Face Surveillance</b></h1>


## Table of Contents
- [Quick Start](#quick-start)
- [Introduction](#introduction)
- [Usage](#usage)
    - [Clone the Github Repo](#clone-the-github-repo)
    - [Install requirements](#install-requirements)
    - [Training for Face Detection](#training-face-detection-model)
    - [Training for Face Recognition](#training-face-recognition-model)
    - [Face Surveillance]
- [References](#references)

## Quick Start
### Video Demo
<p align="center"> 
  <img src="out/test_video.gif" alt="Sample signal" width="70%" height="10%">
</p>

Note that in this video demo, I haven't created embeddings for Marques Brownlee face, thus his face is labeled as **unknown**.

### Camera Demo
<p align="center"> 
  <img src="out/test_camera.gif" alt="Sample signal" width="70%" height="10%">
</p>

## **Introduction**
This project focuses on developing a face surveillance system. By integrating face detection, recognition, and tracking, this project aim to create a comprehensive pipeline for efficient and accurate facial analysis. The project also includes training and testing scripts for both face detection and recognition models, enabling customization and improvement of performance.

## **Usage**

### Clone the Github Repo
```bash
git clone https://github.com/TaiQuach123/Face_Surveillance.git
```


### Install requirements
```bash
pip install requirements.txt
```

### Training Face Detection model
To train your face detection model, run the following script: 

```bash
python train_detection.py
```

You can adjust arguments and configs such as network architecture, batch size, lr,... 
See example at ``face_detection_train.ipynb`` in ``notebooks``.

### Training Face Recognition model
To train your face recognition model, run the following script:

```bash
python train_recognition.py
```
You can adjust arguments and configs such as network architecture, batch size, lr,... See example at ``face_recognition_train.ipynb`` in ``notebooks``.

Note that for training face recognition model, you can combine different metrics (ArcFace, SubCenter ArcFace, Sphere, ...) with different loss (CrossEntropy, FocalLoss, LabelSmooth,...). Depending on your architecture and data, those combinations may have different results.

### Face Surveillance
To run Face Surveillance system, run the following code:
```bash
python core.py
```

## **References**
[1] https://github.com/ronghuaiyang/arcface-pytorch

[2] https://github.com/deepinsight/insightface

[3] https://github.com/biubug6/Pytorch_Retinaface

[4] https://github.com/HamadYA/GhostFaceNets
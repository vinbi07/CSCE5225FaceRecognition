# Face Recognition Project

This project detects faces from a live webcam or saved video, compares them against a registered set of faces, and labels each detection as either a known person or a stranger.

## Project Contents

- `main.m` - live webcam face recognition and stranger video logging
- `algoTraining.m` - builds `trainedFaceModel.mat` from the images in `trainingImages/`
- `evaluateVideoRecognition.m` - tests the recognizer on a saved video in `testVideos/`
- `showTrainingPipeline.m` - visualizes the preprocessing and face extraction pipeline on one image
- `showModelMontage.m` - displays the processed face samples stored in the trained model
- `trainingImages/` - labeled training images organized by person name
- `testVideos/` - sample test videos
- `trainedFaceModel.mat` - saved trained model used by `main.m` and evaluation scripts

## MATLAB Requirements

This project was developed in MATLAB and uses functions from:

- Computer Vision Toolbox
- Image Processing Toolbox
- Webcam support for MATLAB (`webcam`), if running `main.m`

If the webcam is not available on the machine, use `evaluateVideoRecognition.m` instead of `main.m`.

## Folder Setup

Keep the following items in the same project folder:

- all `.m` files
- `trainedFaceModel.mat`
- `trainingImages/`
- `testVideos/`

`main.m`, `algoTraining.m`, and `evaluateVideoRecognition.m` all use paths relative to the project folder.

## How To Run

Open MATLAB, change the Current Folder to this project directory, then use one of the following entry points.

### 1. Run the live webcam recognizer

```matlab
main
```

What it does:

- opens the default webcam
- detects faces in each frame
- matches faces against the trained model
- labels detections as a registered person or `Stranger`
- records short clips of stranger events into `strangerVideoLogs/`

How to stop:

- press `Esc`
- press `Q`
- or click the on-screen `Stop` button

### 2. Rebuild the trained model

If `trainedFaceModel.mat` is missing or you want to retrain the recognizer:

```matlab
algoTraining
```

This script scans `trainingImages/`, extracts face features, applies simple augmentation, and saves the trained model as:

```matlab
trainedFaceModel.mat
```

Important note:

- each person should have their own folder inside `trainingImages/`
- the folder name is used as that person's label

### 3. Evaluate on a saved video

To test the recognizer on a sample video:

```matlab
report = evaluateVideoRecognition('testVideos/joeyTest.mp4', 'Joey');
```

This function prints a summary including:

- frames processed
- frames with a detected face
- correct recognitions
- stranger detections
- overall accuracy

### 4. Visualize the training pipeline

To see how one image is preprocessed before feature extraction:

```matlab
showTrainingPipeline
```

Or pass a specific image:

```matlab
showTrainingPipeline('trainingImages/Joey/joeyFront_200x267.png')
```

### 5. View the trained face montage

To browse the processed faces saved in the model:

```matlab
showModelMontage
```

## Expected Output

- Live webcam recognition window from `main.m`
- Console output showing best match, distance, confidence, and result
- Logged stranger clips written to `strangerVideoLogs/`
- Evaluation summary printed by `evaluateVideoRecognition.m`

## Troubleshooting

- If `main.m` says the model file is missing, run `algoTraining` first.
- If the webcam cannot be opened, use `evaluateVideoRecognition` with a video file.
- If no faces are found during training, make sure the images in `trainingImages/` contain clear visible faces.
- If MATLAB reports missing toolbox functions such as `vision.CascadeObjectDetector`, `extractLBPFeatures`, or `adapthisteq`, install the required toolboxes listed above.

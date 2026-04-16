% Addison Nally
% main.m
% Live camera face detection

clear;
clc;
close all;

cam = webcam;

targetSize = [200 200];

% Joey Suliguin: Load trained model from algoTraining
modelPath = fullfile(fileparts(mfilename('fullpath')), 'trainedFaceModel.mat');
if ~isfile(modelPath)
    error('Model file not found. Run algoTraining first.');
end

modelData = load(modelPath, 'model');
trainedFeatures = double(modelData.model.features);
trainedLabels = string(modelData.model.labels);

if isempty(trainedFeatures)
    error('Model is empty. Re-run algoTraining.');
end

% PLACEHOLDER: Change if too many strangers are marked as registered (or vice versa).
% FIX: Calculate Distance
distanceThreshold = 35;

desiredFPS = 5;
pauseTime = 1 / desiredFPS;

figure;

while ishandle(gcf)
    % Get frame from camera
    frame = snapshot(cam);

    % Get frame from preprocessing
    processedFrame = preProcess(frame, targetSize);

    % Detect faces in the processed frame
    boundingBox = detectFace(processedFrame);

    % Scale bounding boxes back to original frame size
    scaleX = size(frame, 2) / targetSize(2);
    scaleY = size(frame, 1) / targetSize(1);

    scaledBox = boundingBox;
    for i = 1:size(boundingBox, 1)
        scaledBox(i,1) = boundingBox(i,1) * scaleX;
        scaledBox(i,2) = boundingBox(i,2) * scaleY;
        scaledBox(i,3) = boundingBox(i,3) * scaleX;
        scaledBox(i,4) = boundingBox(i,4) * scaleY;
    end

    % Draw results
    if ~isempty(scaledBox)
        labels = strings(size(scaledBox,1), 1);

        if size(frame, 3) == 3
            grayFrame = rgb2gray(frame);
        else
            grayFrame = frame;
        end

        for i = 1:size(scaledBox, 1)
            box = scaledBox(i, :);
            lbpVector = extractLBPFromBox(grayFrame, box, targetSize);

            if isempty(lbpVector)
                labels(i) = "Stranger";
                continue;
            end

            [bestDistance, bestIdx] = min(pdist2(double(lbpVector), trainedFeatures, 'euclidean'));

            if bestDistance <= distanceThreshold
                labels(i) = "Registered: " + trainedLabels(bestIdx);
            else
                labels(i) = "Stranger";
            end
        end

        outFrame = insertShape(frame, 'Rectangle', scaledBox, ...
            'Color', 'green', 'LineWidth', 3);
        outFrame = insertText(outFrame, scaledBox(:,1:2), labels, ...
            'BoxColor', 'yellow', 'FontSize', 18);
    else
        outFrame = frame;
    end

    imshow(outFrame);
    pause(pauseTime);
    title('Live Face Detection - Press Ctrl+C in Command Window to stop');
    drawnow;
end

clear cam;

function lbpVector = extractLBPFromBox(grayFrame, box, targetSize)
x = max(1, floor(box(1)));
y = max(1, floor(box(2)));
w = max(1, floor(box(3)));
h = max(1, floor(box(4)));

x2 = min(size(grayFrame, 2), x + w - 1);
y2 = min(size(grayFrame, 1), y + h - 1);

if x2 <= x || y2 <= y
    lbpVector = [];
    return;
end

faceCrop = grayFrame(y:y2, x:x2);
processedFace = preProcess(faceCrop, targetSize);
lbpVector = extractLBPFeatures(processedFace);
end

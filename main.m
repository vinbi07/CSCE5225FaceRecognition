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
distanceThreshold = 0.4; 

desiredFPS = 15;
pauseTime = 1 / (2*desiredFPS); % multiply by 2 to account for processing time

figure;

while ishandle(gcf)
    % Get frame from camera
    frame = snapshot(cam);

    % convert the frame to grayscale, and use this frame to detect the faces
    % and assign labels
    if size(frame, 3) == 3
        grayFrame = rgb2gray(frame);
    else
        grayFrame = frame;
    end

    % downsample so that the calculations can be run faster
    % using full images for detection is too slow
    detectionScale = 0.2; % change this if its laggy, lower = more accurate
    smallGray = imresize(grayFrame, detectionScale);
    boundingBox = detectFace(smallGray);

    scaledBox = boundingBox;
    if ~isempty(boundingBox)
        scaledBox(:, 1:4) = round(boundingBox(:, 1:4) / detectionScale);
    end

    % Draw results
    if ~isempty(scaledBox)
        labels = strings(size(scaledBox,1), 1);

        for i = 1:size(scaledBox, 1)
            box = scaledBox(i, :);
            lbpVector = algoProcess(grayFrame, box, targetSize);

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

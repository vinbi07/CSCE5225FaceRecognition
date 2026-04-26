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

% Dynamic live metrics (no manual label input required).
liveEval.totalDetections = 0;
liveEval.registeredDetections = 0;
liveEval.strangerDetections = 0;
liveEval.frameCount = 0;

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

    % Apply adaptive histogram equalization for possible lighting issues
    normGray = adapthisteq(grayFrame, 'ClipLimit', 0.01, 'NumTiles', [8 8]);


    % downsample so that the calculations can be run faster
    detectionScale = 0.2; % change this if its laggy, lower = more accurate
    smallGray = imresize(grayFrame, detectionScale);
    boundingBox = detectFace(smallGray);

    scaledBox = boundingBox;
    if ~isempty(boundingBox)
        scaledBox(:, 1:4) = round(boundingBox(:, 1:4) / detectionScale);
    end

    frameSummary = 'This Frame | Faces: 0 | Registered: 0 | Stranger: 0';

    % Draw results
    if ~isempty(scaledBox)
        labels = strings(size(scaledBox,1), 1);
        isRegistered = false(size(scaledBox,1), 1);

        for i = 1:size(scaledBox, 1)
            box = scaledBox(i, :);
            lbpVector = algoProcess(grayFrame, box, targetSize);

            if isempty(lbpVector)
                labels(i) = "Stranger";
                continue;
            end

            [bestDistance, bestIdx] = min(pdist2(double(lbpVector), trainedFeatures, 'euclidean'));
            confidence = max(0, 1 - (bestDistance / distanceThreshold));

            if bestDistance <= distanceThreshold
                isRegistered(i) = true;
                labels(i) = "Registered: " + trainedLabels(bestIdx) + " | confidence: " + sprintf('%.0f%%', confidence * 100);
            else
                labels(i) = "Stranger";
            end
        end

        % Update cumulative live metrics from all detections in this frame.
        detectionsThisFrame = size(scaledBox, 1);
        registeredThisFrame = nnz(isRegistered);
        strangerThisFrame = detectionsThisFrame - registeredThisFrame;

        liveEval.totalDetections = liveEval.totalDetections + detectionsThisFrame;
        liveEval.registeredDetections = liveEval.registeredDetections + registeredThisFrame;
        liveEval.strangerDetections = liveEval.strangerDetections + strangerThisFrame;

        outFrame = insertShape(frame, 'Rectangle', scaledBox, ...
            'Color', 'green', 'LineWidth', 3);
        outFrame = insertText(outFrame, scaledBox(:,1:2), labels, ...
            'BoxColor', 'yellow', 'FontSize', 18);

        frameSummary = sprintf('This Frame | Faces: %d | Registered: %d | Stranger: %d', ...
            detectionsThisFrame, registeredThisFrame, strangerThisFrame);
    else
        outFrame = frame;
    end

    outFrame = insertText(outFrame, [10 10], frameSummary, ...
        'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 16);

    liveEval.frameCount = liveEval.frameCount + 1;

    if liveEval.totalDetections > 0
        runningAccuracy = 100 * (liveEval.registeredDetections / liveEval.totalDetections);
        statusText = sprintf('Cumulative | Frames: %d | Detections: %d | Registered Rate: %.2f%%', ...
            liveEval.frameCount, liveEval.totalDetections, runningAccuracy);
        outFrame = insertText(outFrame, [10 40], statusText, ...
            'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 16);
    end

    imshow(outFrame);
    pause(pauseTime);
    title('Live Face Detection - Press Ctrl+C in Command Window to stop');
    drawnow;
end

clear cam;

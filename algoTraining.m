function model = algoTraining(trainingDir, outputMatPath)

% Input Arguments:
if nargin < 1 || isempty(trainingDir)
    trainingDir = fullfile(fileparts(mfilename('fullpath')), 'trainingImages');
end

if nargin < 2 || isempty(outputMatPath)
    outputMatPath = fullfile(fileparts(mfilename('fullpath')), 'trainedFaceModel.mat');
end

% Size for face crops for better processing
targetSize = [200 200];

% Load training images
imds = imageDatastore(trainingDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imageFiles = numel(imds.Files);
if imageFiles == 0
    error('No training images found in: %s', trainingDir);
end

% Make Storage for features and labels
features = [];
trainedLabels = categorical.empty(0, 1);
previewFaces = {};
previewFaceLabels = strings(0, 1);
skippedNoFace = 0;

% Loop for getting faces and features
for i = 1:imageFiles
    frame = readimage(imds, i);

    % Pre Process for Detection
    detectionFrame = prepareDetectionFrame(frame);
    boxes = detectFace(detectionFrame);

    if isempty(boxes)
        skippedNoFace = skippedNoFace + 1;
        continue;
    end

    % Grabbign the largest face from detectface.m
    areas = boxes(:, 3) .* boxes(:, 4);
    [~, idx] = max(areas);
    faceBox = boxes(idx, :);

    faceCrop = imcrop(detectionFrame, faceBox);

    if isempty(faceCrop)
        skippedNoFace = skippedNoFace + 1;
        continue;
    end

    % Pre Process before feature extraction
    processedFace = preProcess(faceCrop, targetSize);
    LBPVector = extractLBPFeatures(processedFace);

    % Storing Data
    features = [features; LBPVector];
    trainedLabels = [trainedLabels; imds.Labels(i)];
    previewFaces{end + 1, 1} = processedFace;
    previewFaceLabels(end + 1, 1) = string(imds.Labels(i)) + " | original";

    % Augementing Data - Flips, Rotations, and Gamma Adjustments
    augs = { ...
        fliplr(faceCrop), ...
        imrotate(faceCrop, -10, 'bilinear', 'crop'), ...
        imrotate(faceCrop,  10, 'bilinear', 'crop'), ...
        imadjust(faceCrop, [], [], 0.8), ...
        imadjust(faceCrop, [], [], 1.2), ...
        };

    for a = 1:numel(augs)
        augCrop = augs{a};
        if isempty(augCrop), continue; end

        augProcessed = preProcess(augCrop, targetSize);
        augLBP = extractLBPFeatures(augProcessed);

        features = [features; augLBP];
        trainedLabels = [trainedLabels; imds.Labels(i)];
        previewFaces{end + 1, 1} = augProcessed;

        % Add Labels for Augmentations
        switch a
            case 1
                augName = "flip";
            case 2
                augName = "rotate -10";
            case 3
                augName = "rotate +10";
            case 4
                augName = "gamma 0.8";
            case 5
                augName = "gamma 1.2";
            otherwise
                augName = "augmented";
        end

        previewFaceLabels(end + 1, 1) = string(imds.Labels(i)) + " | " + augName;
    end
end

if isempty(features)
    error('No valid faces were extracted. Check training images and detector settings.');
end

% Model Structure
model = struct();
model.targetSize = targetSize;
model.features = features;
model.labels = trainedLabels;
model.previewFaces = previewFaces;
model.previewLabels = previewFaceLabels;

labelCats = categorical(trainedLabels);
faceNames = categories(labelCats);
faceCounts = countcats(labelCats);
hasSamples = faceCounts > 0;

model.registeredFaces = string(faceNames(hasSamples));
model.samplesPerFace = faceCounts(hasSamples);
model.recognition = buildRecognitionConfig(double(features), string(trainedLabels));

% Save the model
save(outputMatPath, 'model');

% Output Summary
fprintf('Training complete.\n');
fprintf('Images scanned: %d\n', imageFiles);
fprintf('Faces learned: %d\n', size(features, 1));
fprintf('Skipped (no face): %d\n', skippedNoFace);
fprintf('Model saved to: %s\n', outputMatPath);
end

% Build reccognition config based on Euclidean distance
function recognition = buildRecognitionConfig(features, trainedLabels)
distanceMatrix = pdist2(features, features, 'euclidean');
sampleCount = size(distanceMatrix, 1);
distanceMatrix(1:sampleCount+1:end) = inf;

nearestSame = [];

% Find closest sample of same person
for i = 1:sampleCount
    sameMask = trainedLabels == trainedLabels(i);
    sameMask(i) = false;

    if any(sameMask)
        nearestSame(end + 1) = min(distanceMatrix(i, sameMask)); %#ok<AGROW>
    end
end

% Calc thresholds based on if it will be in the 90th percentile
if isempty(nearestSame)
    distanceThreshold = 0.60; % In case someone only has one pic
else
    distanceThreshold = prctile(nearestSame, 90);
end

confidenceMinDistance = minOrFallback(nearestSame, 0);
confidenceMaxDistance = maxOrFallback(nearestSame, distanceThreshold);

% Save
recognition = struct();
recognition.distanceThreshold = max(distanceThreshold, 0.05);
recognition.confidenceMinDistance = max(confidenceMinDistance, 0);
recognition.confidenceMaxDistance = max(confidenceMaxDistance, recognition.confidenceMinDistance + 0.05);
end

% Function for calc min or fallback value
function value = minOrFallback(values, fallbackValue)
if isempty(values)
    value = fallbackValue;
else
    value = min(values);
end
end

function value = maxOrFallback(values, fallbackValue)
if isempty(values)
    value = fallbackValue;
else
    value = max(values);
end
end


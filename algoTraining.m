% Joey Suliguin: Model for training for Face Recognition
function model = algoTraining(trainingDir, outputMatPath)

% Get Paths
if nargin < 1 || isempty(trainingDir)
    trainingDir = fullfile(fileparts(mfilename('fullpath')), 'trainingImages');
end

if nargin < 2 || isempty(outputMatPath)
    outputMatPath = fullfile(fileparts(mfilename('fullpath')), 'trainedFaceModel.mat');
end


% Crop Size
targetSize = [200 200];

% Initialize the face detector
detector = vision.CascadeObjectDetector('FrontalFaceCART');

% Point to the folder containing subfolders for each person
imds = imageDatastore(trainingDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Get Image Files
imageFiles = numel(imds.Files);
if imageFiles == 0
    error('No training images found in: %s', trainingDir);
end

% Arrays to hold features and labels
features = [];
trainedLabels = [];
skippedNoFace = 0;

% Go through the images
for i = 1:imageFiles
    frame = readimage(imds, i);

    % Check if Gray if not convert to gray
    if size(frame, 3) == 3
        grayFrame = rgb2gray(frame);
    else
        grayFrame = frame;
    end

    % Detect the faces in the image
    boxes = detector(grayFrame);

    % If no face, skip and +1 to skippedNoFace
    if isempty(boxes)
        skippedNoFace = skippedNoFace + 1;
        continue;
    end

    % Use the largest detected face for crop
    areas = boxes(:, 3) .* boxes(:, 4);
    [~, idx] = max(areas);
    faceBox = boxes(idx, :);

    faceCrop = imcrop(grayFrame, faceBox);

    % If crop empty, skip and +1 to skippedNoFace
    if isempty(faceCrop)
        skippedNoFace = skippedNoFace + 1;
        continue;
    end

    % Process and extract LBP features
    processedFace = preProcess(faceCrop, targetSize);
    LBPVector = extractLBPFeatures(processedFace);

    % Append the new vector as a new row
    features = [features; LBPVector];
    trainedLabels = [trainedLabels; imds.Labels(i)];
end

% Check if any features were extracted
if isempty(features)
    error('No valid faces were extracted. Check training images and detector settings.');
end

% Save the model & Get Results
model = struct();
model.targetSize = targetSize;
model.features = features;
model.labels = trainedLabels;

save(outputMatPath, 'model');

fprintf('Training complete.\n');
fprintf('Images scanned: %d\n', imageFiles);
fprintf('Faces learned: %d\n', size(features, 1));
fprintf('Skipped (no face): %d\n', skippedNoFace);
fprintf('Model saved to: %s\n', outputMatPath);
end


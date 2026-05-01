function showTrainingPipeline(imagePath, targetSize)

% Input Arguments
if nargin < 1 || isempty(imagePath)
    trainingDir = fullfile(fileparts(mfilename('fullpath')), 'trainingImages');
    imageFiles = dir(fullfile(trainingDir, '**', '*.*'));
    imageFiles = imageFiles(~[imageFiles.isdir]);

    if isempty(imageFiles)
        error('No training images found in: %s', trainingDir);
    end

    imagePath = fullfile(imageFiles(1).folder, imageFiles(1).name);
end

if nargin < 2 || isempty(targetSize)
    targetSize = [200 200];
end

if ~isfile(imagePath)
    error('Image file not found: %s', imagePath);
end

% Get the frame/image to run through the pipeline
frame = imread(imagePath);
[detectionFrame, detectionSteps] = prepareDetectionFrame(frame);

boxes = detectFace(detectionFrame);

if isempty(boxes)
    error('No face detected in image: %s', imagePath);
end

% Grab Biggest face
areas = boxes(:, 3) .* boxes(:, 4);
[~, idx] = max(areas);
faceBox = boxes(idx, :);
faceCrop = imcrop(detectionFrame, faceBox);

if isempty(faceCrop)
    error('Detected face crop is empty for image: %s', imagePath);
end

% Process before feature extraction
[processedFace, debugSteps] = preProcess(faceCrop, targetSize);
previewPadding = 12;
paddedDetectionFrame = padarray(detectionFrame, [previewPadding previewPadding], 'replicate', 'both');
paddedFaceBox = faceBox + [previewPadding previewPadding 0 0];
labelPosition = [paddedFaceBox(1), max(1, paddedFaceBox(2) - 25)];

% UI to show steps
fig = figure('Name', 'Algo Training Pipeline', 'NumberTitle', 'off');
tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
% Original Image
imshow(frame);
title('Original Image', 'Interpreter', 'none');

nexttile;
% Greyscale
imshow(detectionSteps.grayFrame);
title('1. Full-Frame Grayscale', 'Interpreter', 'none');

nexttile;
% Adaptive Histogram Equalization
imshow(detectionSteps.equalizedFrame);
title('2. Adaptive Equalized Frame', 'Interpreter', 'none');

nexttile;
% Denoise
imshow(detectionSteps.denoisedFrame);
title('3. Denoised Frame (used by detectFace)', 'Interpreter', 'none');

nexttile;
% Detected Face with Bounding Box
imshow(paddedDetectionFrame);
hold on;
rectangle('Position', paddedFaceBox, 'EdgeColor', 'g', 'LineWidth', 3);
text(labelPosition(1), labelPosition(2), 'Largest Face', ...
    'Color', 'black', 'BackgroundColor', 'green', 'FontSize', 12, ...
    'Margin', 2, 'Interpreter', 'none');
hold off;
title('4. Largest Returned Face Selected', 'Interpreter', 'none');

% Cropped Face
nexttile;
imshow(faceCrop);
title('5. Cropped Equalized Face', 'Interpreter', 'none');

nexttile;
% Resized 200x200
imshow(debugSteps.resizedFrame);
title(sprintf('6. Resized %dx%d', targetSize(1), targetSize(2)), 'Interpreter', 'none');

nexttile;
% Gamma Correction
imshow(debugSteps.gammaNormalized);
title('7. Gamma Corrected', 'Interpreter', 'none');

nexttile;
% Processed face before LBP extraction
imshow(processedFace);
title('8. Input to extractLBPFeatures', 'Interpreter', 'none');

sgtitle(sprintf('Algo Training Pipeline: %s', imagePath), 'Interpreter', 'none');
end

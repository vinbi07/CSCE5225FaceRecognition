function showTrainingPipeline(imagePath, targetSize)

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

frame = imread(imagePath);
[detectionFrame, detectionSteps] = prepareDetectionFrame(frame);

boxes = detectFace(detectionFrame);

if isempty(boxes)
    error('No face detected in image: %s', imagePath);
end

areas = boxes(:, 3) .* boxes(:, 4);
[~, idx] = max(areas);
faceBox = boxes(idx, :);
faceCrop = imcrop(detectionFrame, faceBox);

if isempty(faceCrop)
    error('Detected face crop is empty for image: %s', imagePath);
end

[processedFace, debugSteps] = preProcess(faceCrop, targetSize);
annotatedFrame = insertShape(detectionFrame, 'Rectangle', faceBox, 'Color', 'green', 'LineWidth', 6);
labelPosition = [faceBox(1), max(1, faceBox(2) - 25)];
annotatedFrame = insertText(annotatedFrame, labelPosition, 'Largest Face', ...
    'BoxColor', 'green', 'TextColor', 'black', 'FontSize', 18);

fig = figure('Name', 'Algo Training Pipeline', 'NumberTitle', 'off');
tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
imshow(frame);
title('Original Image', 'Interpreter', 'none');

nexttile;
imshow(detectionSteps.grayFrame);
title('1. Full-Frame Grayscale', 'Interpreter', 'none');

nexttile;
imshow(detectionSteps.equalizedFrame);
title('2. Adaptive Equalized Frame', 'Interpreter', 'none');

nexttile;
imshow(detectionSteps.denoisedFrame);
title('3. Denoised Frame (used by detectFace)', 'Interpreter', 'none');

nexttile;
imshow(annotatedFrame);
title('4. Largest Returned Face Selected', 'Interpreter', 'none');

nexttile;
imshow(faceCrop);
title('5. Cropped Equalized Face', 'Interpreter', 'none');

nexttile;
imshow(debugSteps.resizedFrame);
title(sprintf('6. Resized %dx%d', targetSize(1), targetSize(2)), 'Interpreter', 'none');

nexttile;
imshow(debugSteps.gammaNormalized);
title('7. Gamma Corrected', 'Interpreter', 'none');

nexttile;
imshow(processedFace);
title('8. Input to extractLBPFeatures', 'Interpreter', 'none');

sgtitle(sprintf('Algo Training Pipeline: %s', imagePath), 'Interpreter', 'none');
annotation(fig, 'textbox', [0.12 0.01 0.78 0.05], ...
    'String', 'The shared detection preprocessing now does grayscale, adaptive histogram equalization, and denoising before detectFace; the cropped face then goes through resize, gamma correction, and min-max normalization.', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

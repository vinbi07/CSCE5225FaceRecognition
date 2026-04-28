function report = evaluateVideoRecognition(videoPath, expectedLabel, maxDurationSec, desiredFPS, smoothingWindowSize, showPreview, printPerFrameDebug)
% evaluateVideoRecognition Evaluate recognition accuracy on a video clip.
%
% Example:
%   report = evaluateVideoRecognition('testVideos/videoPlaceholder.mp4', "Joey");

if nargin < 1 || isempty(videoPath)
    videoPath = fullfile(fileparts(mfilename('fullpath')), 'testVideos', 'videoPlaceholder.mp4');
end

if nargin < 2 || isempty(expectedLabel)
    expectedLabel = "Joey";
end

if nargin < 3 || isempty(maxDurationSec)
    maxDurationSec = 30;
end

if nargin < 4 || isempty(desiredFPS)
    desiredFPS = 10;
end

if nargin < 5 || isempty(smoothingWindowSize)
    smoothingWindowSize = 5;
end

if nargin < 6 || isempty(showPreview)
    showPreview = true;
end

if nargin < 7 || isempty(printPerFrameDebug)
    printPerFrameDebug = true;
end

expectedLabel = string(expectedLabel);

if ~isfile(videoPath)
    error('Video file not found: %s', videoPath);
end

targetSize = [200 200];
detectionScale = 0.18;

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

recognition = [];
if isfield(modelData.model, 'recognition')
    recognition = modelData.model.recognition;
end
recognition = ensureRecognitionConfig(recognition);

videoReader = VideoReader(videoPath);
maxProcessTime = min(maxDurationSec, videoReader.Duration);
sampleInterval = 1 / desiredFPS;
nextSampleTime = 0;

labelQueue = strings(0, 1);
report = initializeReport(videoPath, expectedLabel, maxProcessTime, desiredFPS, smoothingWindowSize);

fig = [];
if showPreview
    fig = figure('Name', 'Video Recognition Evaluation', 'NumberTitle', 'off');
end

while hasFrame(videoReader) && videoReader.CurrentTime <= maxProcessTime
    frame = readFrame(videoReader);
    frameTime = videoReader.CurrentTime;

    if frameTime + eps < nextSampleTime
        continue;
    end
    nextSampleTime = nextSampleTime + sampleInterval;

    report.framesProcessed = report.framesProcessed + 1;

    detectionFrame = prepareDetectionFrame(frame);
    smallGray = imresize(detectionFrame, detectionScale);
    boundingBox = detectFace(smallGray);

    scaledBox = boundingBox;
    if ~isempty(boundingBox)
        scaledBox(:, 1:4) = round(boundingBox(:, 1:4) / detectionScale);
    end

    currentDisplay = frame;
    smoothedLabel = "No Face";
    currentConfidence = 0;
    currentDistance = inf;

    if ~isempty(scaledBox)
        areas = scaledBox(:, 3) .* scaledBox(:, 4);
        [~, idx] = max(areas);
        faceBox = scaledBox(idx, :);
        report.framesWithFace = report.framesWithFace + 1;

        lbpVector = algoProcess(detectionFrame, faceBox, targetSize);

        if isempty(lbpVector)
            rawLabel = "Stranger";
            bestLabel = "No Match";
            isRegistered = false;
        else
            matchResult = matchRegisteredFace(double(lbpVector), trainedFeatures, trainedLabels, recognition);
            rawLabel = ternary(matchResult.isRegistered, matchResult.label, "Stranger");
            bestLabel = matchResult.label;
            isRegistered = matchResult.isRegistered;
            currentConfidence = matchResult.confidence;
            currentDistance = matchResult.distance;
        end

        if printPerFrameDebug
            fprintf(['Frame %d | Best label: %s | Distance: %.4f | Threshold: %.4f | ' ...
                'Result: %s | Smoothed: %s\n'], ...
                report.framesProcessed, ...
                bestLabel, ...
                currentDistance, ...
                recognition.distanceThreshold, ...
                string(ternary(isRegistered, "Registered", "Stranger")), ...
                rawLabel);
        end

        labelQueue(end + 1, 1) = rawLabel;
        if numel(labelQueue) > smoothingWindowSize
            labelQueue(1) = [];
        end

        smoothedLabel = computeModeLabel(labelQueue);
        if printPerFrameDebug
            fprintf('Frame %d | Queue mode result: %s\n', report.framesProcessed, smoothedLabel);
        end
        report = updateReportCounts(report, smoothedLabel, expectedLabel, currentConfidence, currentDistance);

        displayLabel = buildDisplayLabel(smoothedLabel, expectedLabel, currentConfidence);
        currentDisplay = insertShape(frame, 'Rectangle', faceBox, 'Color', 'green', 'LineWidth', 3);
        currentDisplay = insertText(currentDisplay, faceBox(1:2), displayLabel, ...
            'BoxColor', 'yellow', 'FontSize', 18);
    else
        report.framesNoFace = report.framesNoFace + 1;
        if printPerFrameDebug
            fprintf('Frame %d | No face detected\n', report.framesProcessed);
        end
    end

    if showPreview && ishandle(fig)
        summaryText = sprintf('Expected: %s | Frame %d | Current: %s', ...
            expectedLabel, report.framesProcessed, smoothedLabel);
        previewFrame = insertText(currentDisplay, [10 10], summaryText, ...
            'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 16);
        imshow(previewFrame, 'Parent', gca);
        title('Video Recognition Evaluation');
        drawnow;
    end
end

report.accuracyOnDetectedFaces = safePercent(report.correctExpectedFrames, report.framesWithFace);
report.accuracyOverall = safePercent(report.correctExpectedFrames, report.framesProcessed);
report.noFaceRate = safePercent(report.framesNoFace, report.framesProcessed);

fprintf('\nVideo Evaluation Summary\n');
fprintf('Video: %s\n', videoPath);
fprintf('Expected label: %s\n', expectedLabel);
fprintf('Processed duration: %.2f sec\n', maxProcessTime);
fprintf('Frames processed: %d\n', report.framesProcessed);
fprintf('Frames with face: %d\n', report.framesWithFace);
fprintf('Correct (%s): %d\n', expectedLabel, report.correctExpectedFrames);
fprintf('Incorrect registered label: %d\n', report.wrongRegisteredFrames);
fprintf('Stranger: %d\n', report.strangerFrames);
fprintf('No face: %d\n', report.framesNoFace);
fprintf('Accuracy on detected faces: %.2f%%\n', report.accuracyOnDetectedFaces);
fprintf('Overall accuracy: %.2f%%\n', report.accuracyOverall);

if showPreview && ~isempty(fig) && ishandle(fig)
    figure(fig);
end
end

function report = initializeReport(videoPath, expectedLabel, maxDurationSec, desiredFPS, smoothingWindowSize)
report = struct();
report.videoPath = string(videoPath);
report.expectedLabel = expectedLabel;
report.maxDurationSec = maxDurationSec;
report.desiredFPS = desiredFPS;
report.smoothingWindowSize = smoothingWindowSize;
report.framesProcessed = 0;
report.framesWithFace = 0;
report.framesNoFace = 0;
report.correctExpectedFrames = 0;
report.wrongRegisteredFrames = 0;
report.strangerFrames = 0;
report.confidenceSumCorrect = 0;
report.distanceSumCorrect = 0;
report.correctConfidenceSamples = 0;
end

function report = updateReportCounts(report, smoothedLabel, expectedLabel, currentConfidence, currentDistance)
if smoothedLabel == expectedLabel
    report.correctExpectedFrames = report.correctExpectedFrames + 1;
    if isfinite(currentConfidence)
        report.confidenceSumCorrect = report.confidenceSumCorrect + currentConfidence;
        report.correctConfidenceSamples = report.correctConfidenceSamples + 1;
    end
    if isfinite(currentDistance)
        report.distanceSumCorrect = report.distanceSumCorrect + currentDistance;
    end
elseif smoothedLabel == "Stranger"
    report.strangerFrames = report.strangerFrames + 1;
else
    report.wrongRegisteredFrames = report.wrongRegisteredFrames + 1;
end
end

function label = computeModeLabel(labelQueue)
if isempty(labelQueue)
    label = "No Face";
    return;
end

[uniqueLabels, ~, idx] = unique(labelQueue);
labelCounts = accumarray(idx(:), 1);
[~, modeIdx] = max(labelCounts);
label = uniqueLabels(modeIdx);
end

function displayLabel = buildDisplayLabel(smoothedLabel, expectedLabel, currentConfidence)
if smoothedLabel == "Stranger"
    displayLabel = "Stranger";
elseif smoothedLabel == expectedLabel
    displayLabel = "Expected: " + smoothedLabel + " | confidence: " + sprintf('%.0f%%', currentConfidence * 100);
else
    displayLabel = "Wrong Label: " + smoothedLabel + " | confidence: " + sprintf('%.0f%%', currentConfidence * 100);
end
end

function recognition = ensureRecognitionConfig(recognition)
if nargin < 1 || isempty(recognition)
    recognition = struct();
end

if ~isfield(recognition, 'distanceThreshold') || isempty(recognition.distanceThreshold) ...
        || ~isfinite(recognition.distanceThreshold) || recognition.distanceThreshold <= 0
    recognition.distanceThreshold = 0.75;
end

if ~isfield(recognition, 'confidenceMinDistance') || isempty(recognition.confidenceMinDistance) ...
        || ~isfinite(recognition.confidenceMinDistance) || recognition.confidenceMinDistance < 0
    recognition.confidenceMinDistance = 0;
end

if ~isfield(recognition, 'confidenceMaxDistance') || isempty(recognition.confidenceMaxDistance) ...
        || ~isfinite(recognition.confidenceMaxDistance) ...
        || recognition.confidenceMaxDistance <= recognition.confidenceMinDistance
    recognition.confidenceMaxDistance = max(recognition.distanceThreshold, recognition.confidenceMinDistance + 0.05);
end
end

function matchResult = matchRegisteredFace(lbpVector, trainedFeatures, trainedLabels, recognition)
sampleDistances = pdist2(lbpVector, trainedFeatures, 'euclidean');
[bestDistance, bestIdx] = min(sampleDistances);
bestLabel = trainedLabels(bestIdx);
isRegistered = bestDistance <= recognition.distanceThreshold;
confidence = computeRangeBasedConfidence(bestDistance, recognition);

matchResult = struct();
matchResult.label = bestLabel;
matchResult.distance = bestDistance;
matchResult.isRegistered = isRegistered;
matchResult.confidence = confidence;
end

function confidence = computeRangeBasedConfidence(bestDistance, recognition)
lowerBound = recognition.confidenceMinDistance;
upperBound = recognition.confidenceMaxDistance;
rangeWidth = max(upperBound - lowerBound, eps);

if bestDistance <= lowerBound
    confidence = 1;
elseif bestDistance >= upperBound
    confidence = 0;
else
    confidence = 1 - ((bestDistance - lowerBound) / rangeWidth);
end
end

function value = ternary(condition, trueValue, falseValue)
if condition
    value = trueValue;
else
    value = falseValue;
end
end

function pct = safePercent(numerator, denominator)
if denominator <= 0
    pct = 0;
else
    pct = 100 * (numerator / denominator);
end
end

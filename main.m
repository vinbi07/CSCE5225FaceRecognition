% Addison Nally, Joey Suliguin, Joshua Shapiro
% main.m
% Live camera face detection

clear;
clc;
close all;

cam = webcam;
cleanupObj = onCleanup(@() cleanupResources());

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

recognition = [];
if isfield(modelData.model, 'recognition')
    recognition = modelData.model.recognition;
end

recognition = ensureRecognitionConfig(recognition);
fprintf('Recognition threshold: %.4f\n', recognition.distanceThreshold);

% Dynamic live metrics (no manual label input required).
liveEval.totalDetections = 0;
liveEval.registeredDetections = 0;
liveEval.strangerDetections = 0;
liveEval.frameCount = 0;

desiredFPS = 10;
pauseTime = 1 / (2*desiredFPS); % multiply by 2 to account for processing time

% Temporal smoothing: rolling queue of raw labels per face slot

% amount of frames to consider for smoothing
smoothingWindowSize = 10;

% maintain a larger queue for each current face slot
labelQueues = {};

% Stranger logging setup
logFolder = fullfile(fileparts(mfilename('fullpath')), 'strangerVideoLogs');
if ~exist(logFolder, 'dir')
    mkdir(logFolder);
end

logFile = fullfile(logFolder, 'strangerLog.csv');
if ~isfile(logFile)
    fid = fopen(logFile, 'w');
    fprintf(fid, 'Timestamp,VideoFile,TriggerStrangerCount,MaxStrangerCount,BestDistance\n');
    fclose(fid);
end

triggerStrangerCount = 0;
maxStrangerCount = 0;
eventBestDistance = inf;
currentVideoName = "";
bestStrangerDistanceThisFrame = inf;
recordDurationSec = 5;
cooldownSec = 0;
isRecording = false;
recordStartTime = [];
lastRecordingEndTime = [];
videoWriter = [];
strangerFrameCount = 0;
strangerFramesNeeded = 5; % number of consecutive frames with strangers before logging (to avoid false positives)

fig = figure( ...
    'Name', 'Live Face Detection', ...
    'NumberTitle', 'off', ...
    'KeyPressFcn', @handleFigureKeyPress, ...
    'CloseRequestFcn', @handleFigureClose);
setappdata(fig, 'stopRequested', false);

uicontrol( ...
    'Style', 'pushbutton', ...
    'String', 'Stop', ...
    'Position', [10 10 80 30], ...
    'Callback', @handleStopButton);



while ishandle(fig) && ~getappdata(fig, 'stopRequested')
    % Get frame from camera
    frame = snapshot(cam);

    % Apply the shared full-frame preprocessing used before detection.
    detectionFrame = prepareDetectionFrame(frame);

    % downsample so that the calculations can be run faster
    detectionScale = 0.18; % change this if its laggy, lower = more accurate
    smallGray = imresize(detectionFrame, detectionScale);
    boundingBox = detectFace(smallGray);

    scaledBox = boundingBox;
    if ~isempty(boundingBox)
        scaledBox(:, 1:4) = round(boundingBox(:, 1:4) / detectionScale);
    end

    frameSummary = 'This Frame | Faces: 0 | Registered: 0 | Stranger: 0';
    strangerThisFrame = 0;
    bestStrangerDistanceThisFrame = inf;

    % Whenever the number of detected faces changes, reset the label queues
    % get # of faces
    newFaceCount = size(scaledBox, 1);

    % allocate label queues
    if newFaceCount ~= numel(labelQueues)
        labelQueues = cell(1, newFaceCount);

        % reset each of the queues
        for k = 1:newFaceCount
            labelQueues{k} = {};
        end
    end

    % Draw results
    if ~isempty(scaledBox)
        labels = strings(size(scaledBox,1), 1);
        isRegistered = false(size(scaledBox,1), 1);

        for i = 1:size(scaledBox, 1)
            box = scaledBox(i, :);
            lbpVector = algoProcess(detectionFrame, box, targetSize);

            % match the LBP vector
            if isempty(lbpVector)
                rawLabel = "Stranger";
                currentConfidence = 0;
                currentDistance = inf;
            else
                matchResult = matchRegisteredFace(double(lbpVector), trainedFeatures, trainedLabels, recognition);
                fprintf('Best label: %s | Distance: %.4f | Confidence: %.0f%% | Result: %s\n', ...
                    matchResult.label, ...
                    matchResult.distance, ...
                    matchResult.confidence * 100, ...
                    string(ternary(matchResult.isRegistered, "Registered", "Stranger")));

                rawLabel = ternary(matchResult.isRegistered, matchResult.label, "Stranger");
                currentConfidence = matchResult.confidence;
                currentDistance = matchResult.distance;
            end

            % push each of the raw labels into the queue for this slot
            labelQueues{i}{end+1} = rawLabel;
            if numel(labelQueues{i}) > smoothingWindowSize
                labelQueues{i}(1) = [];
            end

            % take the mode of the queue as the smoothed label
            allQueueLabels = string(labelQueues{i});

            % find the # ids for each label
            [uniqueQueueLabels, ~, qIdx] = unique(allQueueLabels);

            % count each label
            labelCounts = accumarray(qIdx(:), 1);
            [~, modeIdx] = max(labelCounts);
            % the mode will be the max of the label counts
            smoothedLabel = uniqueQueueLabels(modeIdx);

            % if not stranger, the new label is the mode
            if smoothedLabel ~= "Stranger"
                isRegistered(i) = true;
                labels(i) = "Registered: " + smoothedLabel + " | confidence: " + ...
                    sprintf('%.0f%%', currentConfidence * 100);
            else
                labels(i) = "Stranger";
                strangerThisFrame = strangerThisFrame + 1;

                if currentDistance < bestStrangerDistanceThisFrame
                    bestStrangerDistanceThisFrame = currentDistance;
                end
            end
        end

        % Update cumulative live metrics from all detections in this frame.
        detectionsThisFrame = size(scaledBox, 1);
        registeredThisFrame = nnz(isRegistered);
        %strangerThisFrame = detectionsThisFrame - registeredThisFrame;

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
        statusText = sprintf('Cumulative | Frames: %d | Detections: %d', ...
            liveEval.frameCount, liveEval.totalDetections);
        outFrame = insertText(outFrame, [10 40], statusText, ...
            'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 16);
    end

    % Stranger logging
    nowTime = datetime('now');

    if strangerThisFrame > 0
        strangerFrameCount = strangerFrameCount + 1;
    else
        strangerFrameCount = 0;
    end

    if strangerFrameCount >= strangerFramesNeeded && ~isRecording
        if isempty(lastRecordingEndTime) || seconds(nowTime - lastRecordingEndTime) >= cooldownSec
            timestampStr = string(nowTime, 'yyyy-MM-dd_HH-mm-ss');
            videoName = "stranger_" + timestampStr + ".mp4";
            videoPath = fullfile(logFolder, char(videoName));

            videoWriter = VideoWriter(videoPath, 'MPEG-4');
            videoWriter.FrameRate = desiredFPS;
            open(videoWriter);

            triggerStrangerCount = strangerThisFrame;
            maxStrangerCount = strangerThisFrame;
            eventBestDistance = bestStrangerDistanceThisFrame;
            currentVideoName = videoName;

            isRecording = true;
            recordStartTime = nowTime;
            strangerFrameCount = 0;
        end
    end

    if isRecording
        maxStrangerCount = max(maxStrangerCount, strangerThisFrame);

        if strangerThisFrame > 0 && bestStrangerDistanceThisFrame < eventBestDistance
            eventBestDistance = bestStrangerDistanceThisFrame;
        end

        writeVideo(videoWriter, outFrame);

        if seconds(nowTime - recordStartTime) >= recordDurationSec
            close(videoWriter);
            isRecording = false;
            lastRecordingEndTime = nowTime;

            if isinf(eventBestDistance)
                eventBestDistance = -1;
            end

            fid = fopen(logFile, 'a');
            fprintf(fid, '%s,%s,%d,%d,%.4f\n', ...
                char(string(recordStartTime, 'yyyy-MM-dd HH:mm:ss')), ...
                char(currentVideoName), ...
                triggerStrangerCount, ...
                maxStrangerCount, ...
                eventBestDistance);
            fclose(fid);
        end
    end

    imshow(outFrame, 'Parent', gca);
    pause(pauseTime);
    title('Live Face Detection - Press Esc, Q, or click Stop to exit');
    drawnow;
end

if isRecording
    close(videoWriter);
end

clear cam;

if ishandle(fig)
    delete(fig);
end

function recognition = ensureRecognitionConfig(recognition)
if nargin < 1 || isempty(recognition)
    recognition = struct();
end

if ~isfield(recognition, 'distanceThreshold') || isempty(recognition.distanceThreshold) ...
        || ~isfinite(recognition.distanceThreshold) || recognition.distanceThreshold <= 0
    recognition.distanceThreshold = 0.85;
end
end

function matchResult = matchRegisteredFace(lbpVector, trainedFeatures, trainedLabels, recognition)
sampleDistances = pdist2(lbpVector, trainedFeatures, 'euclidean');
[bestDistance, bestIdx] = min(sampleDistances);
bestLabel = trainedLabels(bestIdx);
isRegistered = bestDistance <= recognition.distanceThreshold;
confidence = max(0, 1 - (bestDistance / recognition.distanceThreshold));

matchResult = struct();
matchResult.label = bestLabel;
matchResult.distance = bestDistance;
matchResult.isRegistered = isRegistered;
matchResult.confidence = confidence;
end

function handleStopButton(~, ~)
fig = gcbf;
if ~isempty(fig) && ishandle(fig)
    setappdata(fig, 'stopRequested', true);
end
end

function handleFigureKeyPress(src, event)
if ismember(lower(event.Key), {'escape', 'q'})
    setappdata(src, 'stopRequested', true);
end
end

function handleFigureClose(src, ~)
setappdata(src, 'stopRequested', true);
end

function cleanupResources()
clear cam;
end

function value = ternary(condition, trueValue, falseValue)
if condition
    value = trueValue;
else
    value = falseValue;
end
end

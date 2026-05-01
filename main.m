% Addison Nally, Joey Suliguin, Joshua Shapiro
% main.m
% Live camera face detection

clear;
clc;
close all;

cam = webcam;
cleanupObj = onCleanup(@() clear('cam')); % Clear cam so webcam doesnt stay on

targetSize = [200 200]; % Face Crop size

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

% Grab The recognition data from Algotraining
recognition = [];
if isfield(modelData.model, 'recognition')
    recognition = modelData.model.recognition;
end

if isempty(recognition)
    recognition = struct();
end

if ~isfield(recognition, 'distanceThreshold') || isempty(recognition.distanceThreshold) ...
        || ~isfinite(recognition.distanceThreshold) || recognition.distanceThreshold <= 0
    recognition.distanceThreshold = 0.8;
end

if ~isfield(recognition, 'confidenceMinDistance') || isempty(recognition.confidenceMinDistance) ...
        || ~isfinite(recognition.confidenceMinDistance) || recognition.confidenceMinDistance < 0
    recognition.confidenceMinDistance = 0;
end

if ~isfield(recognition, 'confidenceMaxDistance') || isempty(recognition.confidenceMaxDistance) ...
        || ~isfinite(recognition.confidenceMaxDistance) ...
        || recognition.confidenceMaxDistance <= recognition.confidenceMinDistance
    recognition.confidenceMaxDistance = recognition.distanceThreshold;
end

% Dynamic live metrics (no manual label input required).
liveEval.totalDetections = 0;
liveEval.registeredDetections = 0;
liveEval.strangerDetections = 0;
liveEval.frameCount = 0;

desiredFPS = 10; % Works better at lower FPS for processing
pauseTime = 1 / (2*desiredFPS); %  Buffer time

% Temporal Label smoothing: rolling queue of raw labels per face slot

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

% Logging variabels for strangers
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

% UI for buttons to stop program
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


% Main Loop for everything
while ishandle(fig) && ~getappdata(fig, 'stopRequested')
    % Get frame from camera
    frame = snapshot(cam);

    % Pre Process for detection
    detectionFrame = prepareDetectionFrame(frame);

    % downsample so that the calculations can be run faster
    detectionScale = 0.25; % scale down to 25% of original size for better processing
    smallGray = imresize(detectionFrame, detectionScale);
    boundingBox = detectFace(smallGray);

    % Reset labels every round
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

    % Draw results for recognition and logging
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
                sampleDistances = pdist2(double(lbpVector), trainedFeatures, 'euclidean');
                [currentDistance, bestIdx] = min(sampleDistances);
                bestLabel = trainedLabels(bestIdx);

                % applying threshold
                if currentDistance <= recognition.distanceThreshold
                    rawLabel = bestLabel;
                    resultText = "Registered";
                else
                    rawLabel = "Stranger";
                    resultText = "Stranger";
                end

                % Calc Confidence
                lowerBound = recognition.confidenceMinDistance;
                upperBound = recognition.confidenceMaxDistance;
                rangeWidth = max(upperBound - lowerBound, eps);

                if currentDistance <= lowerBound
                    currentConfidence = 1;
                elseif currentDistance >= upperBound
                    currentConfidence = 0;
                else
                    currentConfidence = 1 - ((currentDistance - lowerBound) / rangeWidth);
                end

                fprintf('Best label: %s | Distance: %.4f | Confidence: %.0f%% | Result: %s\n', ...
                    bestLabel, ...
                    currentDistance, ...
                    currentConfidence * 100, ...
                    resultText);
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

    % Display metrics on the frame
    outFrame = insertText(outFrame, [10 10], frameSummary, ...
        'BoxColor', 'black', 'TextColor', 'white', 'FontSize', 16);

    liveEval.frameCount = liveEval.frameCount + 1;

    if liveEval.totalDetections > 0
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

    % If Stranger for more than defined number of frames, start logging
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

    % if recording strangers, log the video and update csv
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

            % Append to CSV
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

    % Display the resulting frame
    imshow(outFrame, 'Parent', gca);
    pause(pauseTime);
    title('Live Face Detection - Press Esc, Q, or click Stop to exit');
    drawnow;
end

% Clean up for videoWriter if still open
if isRecording
    close(videoWriter);
end

clear cam;

% Close the figure if it's still open
if ishandle(fig)
    delete(fig);
end

% Stop Button Callback
function handleStopButton(~, ~)
fig = gcbf;
if ~isempty(fig) && ishandle(fig)
    setappdata(fig, 'stopRequested', true);
end
end

% Other Callbacks for escape keys or close button
function handleFigureKeyPress(src, event)
if ismember(lower(event.Key), {'escape', 'q'})
    setappdata(src, 'stopRequested', true);
end
end

function handleFigureClose(src, ~)
setappdata(src, 'stopRequested', true);
end

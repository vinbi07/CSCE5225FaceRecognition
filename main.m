% Addison Nally
% main.m
% Live camera face detection

clear;
clc;
close all;

cam = webcam;

targetSize = [200 200];

desiredFPS = 5;
pauseTime = 1 / desiredFPS;

figure;

while true
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
        outFrame = insertShape(frame, 'Rectangle', scaledBox, ...
            'Color', 'green', 'LineWidth', 3);

        labels = repmat("Stranger", size(scaledBox,1), 1);
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
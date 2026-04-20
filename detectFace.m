function boundingBox = detectFace(frame)
% detectFace.m - Detect multiple faces in an image

detector = vision.CascadeObjectDetector('FrontalFaceCART');


detector.MinSize = [30 30];      % Detect faces as small as 30x30 pixels
detector.MergeThreshold = 4;     % Merge nearby detections for multiple faces

% Detect all faces in the frame
boundingBox = detector(frame);
end
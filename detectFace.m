function boundingBox = detectFace(frame)
% detectFace.m
% Detect frontal faces, side faces, and slightly tilted faces

frontalDetector = vision.CascadeObjectDetector('FrontalFaceCART');
profileDetector = vision.CascadeObjectDetector('ProfileFace');

frontalDetector.MinSize = [30 30];
frontalDetector.MergeThreshold = 4;

profileDetector.MinSize = [30 30];
profileDetector.MergeThreshold = 4;

boundingBox = [];

% Detect frontal faces
frontBoxes = frontalDetector(frame);
boundingBox = [boundingBox; frontBoxes];

% Detect side/profile faces
sideBoxes = profileDetector(frame);
boundingBox = [boundingBox; sideBoxes];

% Detect opposite side by flipping image
flipFrame = fliplr(frame);
flipBoxes = profileDetector(flipFrame);

if ~isempty(flipBoxes)
    imgWidth = size(frame, 2);
    for i = 1:size(flipBoxes, 1)
        flipBoxes(i,1) = imgWidth - flipBoxes(i,1) - flipBoxes(i,3);
    end
    boundingBox = [boundingBox; flipBoxes];
end

% Detect slight head tilt
leftTilt = imrotate(frame, 15, 'crop');
rightTilt = imrotate(frame, -15, 'crop');

leftBoxes = frontalDetector(leftTilt);
rightBoxes = frontalDetector(rightTilt);

boundingBox = [boundingBox; leftBoxes; rightBoxes];

% Remove overlapping boxes
if ~isempty(boundingBox)
    scores = ones(size(boundingBox,1),1);
    boundingBox = selectStrongestBbox(boundingBox, scores, ...
        'OverlapThreshold', 0.3);
end

end
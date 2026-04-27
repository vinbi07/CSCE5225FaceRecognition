function [outFrame, debugSteps] = prepareDetectionFrame(inFrame)
% prepareDetectionFrame Shared preprocessing before face detection.

if nargin < 1 || isempty(inFrame)
    error('prepareDetectionFrame requires one input frame.');
end

debugSteps = struct();

if size(inFrame, 3) == 3
    grayFrame = rgb2gray(inFrame);
else
    grayFrame = inFrame;
end
debugSteps.grayFrame = grayFrame;

equalizedFrame = adapthisteq(grayFrame, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
debugSteps.equalizedFrame = equalizedFrame;

denoisedFrame = medfilt2(equalizedFrame, [3 3]);
debugSteps.denoisedFrame = denoisedFrame;

outFrame = denoisedFrame;
end

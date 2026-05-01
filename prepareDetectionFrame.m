function [outFrame, debugSteps] = prepareDetectionFrame(inFrame)

% Input Arguments
if nargin < 1 || isempty(inFrame)
    error('prepareDetectionFrame requires one input frame.');
end

debugSteps = struct();

% Grayscale
if size(inFrame, 3) == 3
    grayFrame = rgb2gray(inFrame);
else
    grayFrame = inFrame;
end
debugSteps.grayFrame = grayFrame;

% Gamma Normalization
gDouble = double(grayFrame) / 255;
gammaNormalized = uint8(gDouble .^ 0.8 * 255);
debugSteps.gammaNormalized = gammaNormalized;

% Adaptive Histogram Equalization
equalizedFrame = adapthisteq(gammaNormalized, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
debugSteps.equalizedFrame = equalizedFrame;

% Denoise
denoisedFrame = medfilt2(equalizedFrame, [3 3]);
debugSteps.denoisedFrame = denoisedFrame;

% output
outFrame = denoisedFrame;
end

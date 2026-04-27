function [outFrame, debugSteps] = preProcess(inFrame, targetSize)
% Addison Nally, Joey Suliguin
% preProcess.m
% Simple preprocessing for face detection

if nargin < 1 || isempty(inFrame)
    error('preProcess requires one input frame.');
end

if nargin < 2 || isempty(targetSize)
    targetSize = [200 200];
end

debugSteps = struct();

% Check if grayscale
if size(inFrame, 3) == 3
    grayFrame = rgb2gray(inFrame);
else
    grayFrame = inFrame;
end
debugSteps.grayFrame = grayFrame;

% Resize to target size
resizedFrame = imresize(grayFrame, targetSize);
debugSteps.resizedFrame = resizedFrame;

gDouble = double(resizedFrame);
% normalize
gDouble = gDouble / 255;

% normalize the image with gamma correction (brightness)
gammaNormalized = uint8(gDouble .^ 0.8 * 255);
debugSteps.gammaNormalized = gammaNormalized;

% Apply Gaussian filtering for noise reduction
%filteredFrame = imgaussfilt(resizedFrame, 1.0);

filteredDouble = double(gammaNormalized);
minVal = min(filteredDouble(:));
maxVal = max(filteredDouble(:));

if maxVal > minVal
    normalizedFrame = (filteredDouble - minVal) / (maxVal - minVal) * 255;
else
    normalizedFrame = zeros(size(filteredDouble));
end

outFrame = uint8(normalizedFrame);
debugSteps.normalizedFrame = outFrame;
end

function outFrame = preProcess(inFrame, targetSize)
% Addison Nally, Joey Suliguin
% preProcess.m
% Simple preprocessing for face detection

if nargin < 1 || isempty(inFrame)
    error('preProcess requires one input frame.');
end

if nargin < 2
    targetSize = [200 200];
end

if size(inFrame, 3) == 3
    grayFrame = rgb2gray(inFrame);
else
    grayFrame = inFrame;
end

% Apply adaptive histogram equalization for lighting robustness
eqFrame = adapthisteq(grayFrame, 'ClipLimit', 0.02, 'NumTiles', [8 8]);

% Resize to target size
resizedFrame = imresize(eqFrame, targetSize);

% Apply Gaussian filtering for noise reduction
filteredFrame = imgaussfilt(resizedFrame, 1.0);

% Normalize intensity range to [0, 255]
filteredDouble = double(filteredFrame);
minVal = double(min(filteredFrame(:)));
maxVal = double(max(filteredFrame(:)));
normalizedFrame = (filteredDouble - minVal) / (maxVal - minVal) * 255;
outFrame = uint8(normalizedFrame);
end
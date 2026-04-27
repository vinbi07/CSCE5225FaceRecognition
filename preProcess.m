function outFrame = preProcess(inFrame, targetSize)
% Addison Nally, Joey Suliguin
% preProcess.m
% Simple preprocessing for face detection

if nargin < 1 || isempty(inFrame)
    error('preProcess requires one input frame.');
end

if nargin < 2 || isempty(targetSize)
    targetSize = [200 200];
end

if size(inFrame, 3) == 3
    grayFrame = rgb2gray(inFrame);
else
    grayFrame = inFrame;
end

% Resize to target size
resizedFrame = imresize(grayFrame, targetSize);

gDouble = double(resizedFrame);
% normalize
gDouble = gDouble / 255;

% normalize the image with gamma correction (brightness)
gammaNormalized = uint8(gDouble .^ 0.8 * 255);

% Apply adaptive histogram equalization for lighting robustness
eqFrame = adapthisteq(gammaNormalized, 'ClipLimit', 0.02, 'NumTiles', [8 8]);

% Apply Gaussian filtering for noise reduction
%filteredFrame = imgaussfilt(resizedFrame, 1.0);

filteredDouble = double(eqFrame);
minVal = min(filteredDouble(:));
maxVal = max(filteredDouble(:));

if maxVal > minVal
    normalizedFrame = (filteredDouble - minVal) / (maxVal - minVal) * 255;
else
    normalizedFrame = zeros(size(filteredDouble));
end

outFrame = uint8(normalizedFrame);
end

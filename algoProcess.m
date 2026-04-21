function lbpVector = algoProcess(grayFrame, box, targetSize)
x = max(1, floor(box(1)));
y = max(1, floor(box(2)));
w = max(1, floor(box(3)));
h = max(1, floor(box(4)));

x2 = min(size(grayFrame, 2), x + w - 1);
y2 = min(size(grayFrame, 1), y + h - 1);

if x2 <= x || y2 <= y
    lbpVector = [];
    return;
end

faceCrop = grayFrame(y:y2, x:x2);
processedFace = preProcess(faceCrop, targetSize);
lbpVector = extractLBPFeatures(processedFace);
end
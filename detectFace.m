function boundingBox = detectFace(frame)
% detectFace.m
    detector = vision.CascadeObjectDetector('FrontalFaceCART');
    boundingBox = detector(frame);
end
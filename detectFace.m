function bbox = detectFace(frame)
% detectFace.m
    detector = vision.CascadeObjectDetector('FrontalFaceCART');
    bbox = detector(frame);
end
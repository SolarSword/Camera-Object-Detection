import sys
import time
import cv2
from detect_model.yolo_net import YOLONet

image_path = str(sys.argv[1])

yolo = YOLONet()

im = cv2.imread(image_path)

yolo.set_image(im)

#tick1 = time.time()

yolo.detect_from_image()

#tick2 = time.time()
#print(tick2 - tick1)

cv2.imshow('result',yolo.final_with_bounding_box)
cv2.waitKey(0)
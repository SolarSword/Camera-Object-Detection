import sys
import time
import cv2
from detect_model.yolo_v3 import YOLONetv3

image_path = str(sys.argv[1])

yolo = YOLONetv3()

im = cv2.imread(image_path)

yolo.set_image(im)

#tick1 = time.time()

yolo.start()

#tick2 = time.time()
#print(tick2 - tick1)

cv2.imshow('result',yolo.img_detection)
cv2.waitKey(0)
# -*- coding:utf-8 -*-
import time

import numpy as np 
import cv2
from PIL import Image as im 
from matplotlib import pyplot as plt 

# my libraries
from detect_model.yolo_v2 import YOLONetv2
from detect_model.yolo_net import YOLONet
import hog_PIL as nhog
from database import my_database 

class Camera_detect():
    '''This class encapsules the video read part and 
       enter port.

       Since the video part is the first part of my system,
       I encapsule them together in one class

       Attributes:
            similarity_threshold: The threshold of similarity of two 
                                feature vectors
            key_frame_threshold: The threshold to determine a key frame
            last_key_flag: A flag to show whether  last frame is a key frame

    '''
    def __init__(self):
        self.similarity_threshold = 0.5
        self.key_frame_threshold = 3.5
        self.last_key_flag = False

    def diffimage(self, image1, image2):
        '''Compute the difference of two images

        Returns:
            The mean difference of all pixel for two images      
        '''
        image1 = image1.astype(np.int)
        image2 = image2.astype(np.int)
        diff = abs(image2 - image1)
        return np.mean(diff.astype(np.uint8))

    def detected_time(self):
        return time.asctime(time.localtime(time.time()))

    def start(self):
        is_on = False
        cap = cv2.VideoCapture(0)
        ret, last_frame = cap.read()
        # set up model object
        yolov2 = YOLONetv2()
        yolo = YOLONet()

        while(cap.isOpened()):
            ret, frame = cap.read()
            if (not ret):
                break
            
            mean_difference = self.diffimage(last_frame, frame)
            if(mean_difference > self.key_frame_threshold and is_on):
                detect_time = self.detected_time()
                yolo.set_image(frame)
                yolov2.set_image(frame)
                yolo.detect_from_image()
                yolov2.start()
                cv2.imshow('yolov2', yolov2.img_detection)
                cv2.imshow('yolov1', yolo.final_with_bounding_box)
               
                if(not self.last_key_flag):
                    self.last_key_flag = True
                    last_key_features = []
                    
                    for detected_object in yolov2.final_objects:
                        try:
                            last_key_features.append(nhog.HOG(im.fromarray(detected_object)).vector)
                        except ValueError:
                            # undo everthing
                            last_key_features = []
                            self.last_key_flag = False
                            break
                else:
                    curr_features = []
                    for frame_object in yolov2.final_objects:
                        try:
                            curr_features.append(nhog.HOG(im.fromarray(frame_object)).vector)  
                        except ValueError:
                            curr_features = []
                            self.last_key_flag = False
                            break
                    compare_times = min(len(last_key_features), len(curr_features))
                    similarities = []
                    for i in range(compare_times):
                        similarities.append(nhog.calc_similarity(last_key_features[i], curr_features[i]))
                        #print("similarity of object %d is %f" % (i, similarities[i]))
                        last_key_features = curr_features
                
            else:
                self.last_key_flag = False
                cv2.imshow('yolov2', frame)
                cv2.imshow('yolov1', frame)

            last_frame = frame
            if (cv2.waitKey(1) &0xFF == ord('q')):
                print("pressed q")
                break
            elif(cv2.waitKey(1) &0xFF == ord('s')):
                print("pressed s")
                is_on = not is_on

        cap.release()
        cv2.destroyAllWindows()


    
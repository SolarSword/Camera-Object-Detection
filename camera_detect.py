# -*- coding:utf-8 -*-
import datetime
import time
import copy
import tkinter as tk
from threading import Thread

import numpy as np 
import cv2
from PIL import Image as im 
#from matplotlib import pyplot as plt 
from PIL import Image, ImageTk

# my libraries
from detect_model.yolo_v3 import YOLONetv3
from detect_model.yolo_v2 import YOLONetv2
from detect_model.yolo_net import YOLONet
import hog_PIL as nhog
from database import Database 

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
        # flag for whether previous frame is a key frame
        # self.is_last_keyframe = False  
        # flags for whether the detection function is on
        self.is_detection_on_1 = False
        self.is_detection_on_2 = False
        self.is_detection_on_3 = False 
        # flags for whether the video is on
        self.is_video_on_0 = True
        self.is_video_on_1 = True
        self.is_video_on_2 = True
        self.is_video_on_3 = True
        # flag for the whole system
        # more like a switch, actually it is a tradeoff 
        self.is_switch_on = True
        self.yolo_v1 = YOLONet()
        self.yolo_v2 = YOLONetv2()
        self.yolo_v3 = YOLONetv3()

        self.db = Database()


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
        '''Current time
        
        Only return the current, but this mathod can be 
        used as to show the detected time.

        Returns:
            The current time, with the format as
            'yyyy-mm-dd HH:MM:SS'
        '''
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def detection_on_click_1(self):
        self.is_detection_on_1 = not self.is_detection_on_1

    def detection_on_click_2(self):
        self.is_detection_on_2 = not self.is_detection_on_2
    
    def detection_on_click_3(self):
        self.is_detection_on_3 = not self.is_detection_on_3

    def video_on_click_0(self):
        self.is_video_on_0 = not self.is_video_on_0

    def video_on_click_1(self):
        self.is_video_on_1 = not self.is_video_on_1

    def video_on_click_2(self):
        self.is_video_on_2 = not self.is_video_on_2

    def video_on_click_3(self):
        self.is_video_on_3 = not self.is_video_on_3    

    def switch_on_click(self):
        self.is_switch_on = not self.is_switch_on 

    def show_frame_lt(self):
        #_, last_frame = self.cap.read()
        while(self.is_switch_on):
            if(self.is_video_on_0):
                try:
                    ret, frame = self.cap.read()
                except BaseException:
                    continue
                if(not ret):
                    continue
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image = img)
                self.video_lt.imgtk = imgtk
                self.video_lt.configure(image = imgtk)
                # last_frame = frame
                # the flicking problem to be solved
                # time.sleep(0.1)

    def show_frame_ld(self):
        _, last_frame = self.cap.read()
        is_last_keyframe = False
        last_key_features = []
        while(self.is_switch_on):
            if(self.is_video_on_1):
                try:
                    _, frame = self.cap.read()
                except BaseException:
                    continue
                mean_difference = self.diffimage(last_frame, frame)
                if(self.is_detection_on_1 and mean_difference > self.key_frame_threshold):
                    self.yolo_v1.set_image(frame)
                    self.yolo_v1.detect_from_image()
                    detected_time = self.detected_time()
                    # compare the objects with those in the last key frame
                    if(not is_last_keyframe):
                        is_last_keyframe = True
                        num_objects = len(self.yolo_v1.final_objects)
                        for i in range(num_objects): 
                            try:
                                last_key_features.append(nhog.HOG(im.fromarray(self.yolo_v1.final_objects[i])).vector)
                                self.db.insert(detected_time, self.yolo_v1.final_classes_names[i],\
                                    'yolo v1', self.yolo_v1.result[i][1]/self.yolo_v1.w_img, self.yolo_v1.result[i][2]/self.yolo_v1.h_img,\
                                        self.yolo_v1.result[i][3]/self.yolo_v1.w_img, self.yolo_v1.result[i][4]/self.yolo_v1.h_img,\
                                            nhog.HOG(im.fromarray(self.yolo_v1.final_objects[i])).vector)
                                self.message_listbox.insert('end', "[%s]yolo v1 store an object tuple into the database" % detected_time)
                            except ValueError:
                                # undo everthing
                                last_key_features = []
                                is_last_keyframe = False
                                break
                    else:
                        curr_features = []
                        for frame_object in self.yolo_v1.final_objects:
                            try:
                                curr_features.append(nhog.HOG(im.fromarray(frame_object)).vector)  
                            except ValueError:
                                curr_features = []
                                is_last_keyframe = False
                                break
                        compare_times = min(len(last_key_features), len(curr_features))
                        similarities = []
                        for i in range(compare_times):
                            similarities.append(nhog.calc_similarity(last_key_features[i], curr_features[i]))
                            # print("similarity of object %d is %f" % (i, similarities[i]))
                            self.message_listbox.insert('end', "[%s]yolo v1 detected %s %d, similarity is %f" % (detected_time, self.yolo_v1.final_classes_names[i], i, similarities[i]))
                            if(similarities[i] < 0.5):
                                self.db.insert(detected_time, self.yolo_v1.final_classes_names[i],\
                                 'yolo v1', self.yolo_v1.result[i][1]/self.yolo_v1.w_img, self.yolo_v1.result[i][2]/self.yolo_v1.h_img,\
                                     self.yolo_v1.result[i][3]/self.yolo_v1.w_img, self.yolo_v1.result[i][4]/self.yolo_v1.h_img,\
                                         nhog.HOG(im.fromarray(self.yolo_v1.final_objects[i])).vector)
                                self.message_listbox.insert('end', "[%s]yolo v1 store an object tuple into the database" % detected_time)
                        # some problems here
                        last_key_features = copy.deepcopy(curr_features)
                    cv2image = cv2.cvtColor(self.yolo_v1.final_with_bounding_box, cv2.COLOR_BGR2RGBA)
                else:
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    is_last_keyframe = False
                    last_key_features = []
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image = img)
                self.video_ld.imgtk = imgtk
                self.video_ld.configure(image = imgtk)
                last_frame = frame
            

    def show_frame_rt(self):
        _, last_frame = self.cap.read()
        is_last_keyframe = False
        last_key_features = []
        while(self.is_switch_on):
            if(self.is_video_on_2):
                try:
                    _, frame = self.cap.read()
                except BaseException:
                    continue
                mean_difference = self.diffimage(last_frame, frame)
                if(self.is_detection_on_2 and mean_difference > self.key_frame_threshold):
                    self.yolo_v2.set_image(frame)
                    self.yolo_v2.start()
                    detected_time = self.detected_time()
                    # compare the objects with those in the last key frame
                    if(not is_last_keyframe):
                        is_last_keyframe = True
                        num_objects = len(self.yolo_v2.final_objects)
                        for i in range(num_objects): # self.yolo_v2.final_objects:
                            try:
                                last_key_features.append(nhog.HOG(im.fromarray(self.yolo_v2.final_objects[i])).vector)
                                self.db.insert(detected_time, self.yolo_v2.final_classes_names[i], \
                                    'yolo v2', self.yolo_v2.final_relative_positions[i][0], self.yolo_v2.final_relative_positions[i][1],\
                                        self.yolo_v2.final_relative_positions[i][2], self.yolo_v2.final_relative_positions[i][3],\
                                            nhog.HOG(im.fromarray(self.yolo_v2.final_objects[i])).vector)
                                self.message_listbox.insert('end', "[%s]yolo v2 store an object tuple into the database" % detected_time)
                            except ValueError:
                                # undo everthing
                                last_key_features = []
                                is_last_keyframe = False
                                break
                    else:
                        curr_features = []
                        for frame_object in self.yolo_v2.final_objects:
                            try:
                                curr_features.append(nhog.HOG(im.fromarray(frame_object)).vector)  
                            except ValueError:
                                curr_features = []
                                is_last_keyframe = False
                                break
                        compare_times = min(len(last_key_features), len(curr_features))
                        similarities = []
                        for i in range(compare_times):
                            similarities.append(nhog.calc_similarity(last_key_features[i], curr_features[i]))
                            # print("similarity of object %d is %f" % (i, similarities[i]))
                            self.message_listbox.insert('end', "[%s]yolo v2 detected %s %d, similarity is %f" % (detected_time, self.yolo_v2.final_classes_names[i], i, similarities[i]))
                            if(similarities[i] < 0.5):
                                self.db.insert(detected_time, self.yolo_v2.final_classes_names[i], \
                                    'yolo v2', self.yolo_v2.final_relative_positions[i][0], self.yolo_v2.final_relative_positions[i][1],\
                                        self.yolo_v2.final_relative_positions[i][2], self.yolo_v2.final_relative_positions[i][3],\
                                            nhog.HOG(im.fromarray(self.yolo_v2.final_objects[i])).vector)
                                self.message_listbox.insert('end', "[%s]yolo v2 store an object tuple into the database" % detected_time)
                        last_key_features = copy.deepcopy(curr_features)
                        cv2image = cv2.cvtColor(self.yolo_v2.img_detection, cv2.COLOR_BGR2RGBA)
                else:
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    is_last_keyframe = False
                    last_key_features = []
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image = img)
                self.video_rt.imgtk = imgtk
                self.video_rt.configure(image = imgtk)
                last_frame = frame
        
    
    def show_frame_rd(self):
        _, last_frame = self.cap.read()
        is_last_keyframe = False
        last_key_features = []
        while(self.is_switch_on):
            if(self.is_video_on_3):
                try:
                    _, frame = self.cap.read()
                except BaseException:
                    continue
                mean_difference = self.diffimage(last_frame, frame)
                if(self.is_detection_on_3 and mean_difference > self.key_frame_threshold):
                    self.yolo_v3.set_image(frame)
                    self.yolo_v3.start()
                    detected_time = self.detected_time()
                    # compare the objects with those in the last key frame
                    if(not is_last_keyframe):
                        is_last_keyframe = True
                        num_objects = len(self.yolo_v3.final_objects)
                        for i in range(num_objects): # self.yolo_v3.final_objects:
                            try:
                                last_key_features.append(nhog.HOG(im.fromarray(self.yolo_v3.final_objects[i])).vector)
                                self.db.insert(detected_time, self.yolo_v3.final_classes_names[i], \
                                    'yolo v3', self.yolo_v3.final_relative_positions[i][0], self.yolo_v3.final_relative_positions[i][1],\
                                        self.yolo_v3.final_relative_positions[i][2], self.yolo_v3.final_relative_positions[i][3],\
                                            nhog.HOG(im.fromarray(self.yolo_v3.final_objects[i])).vector)
                                self.message_listbox.insert('end', "[%s]yolo v3 store an object tuple into the database" % detected_time)
                            except ValueError:
                                # undo everthing
                                last_key_features = []
                                is_last_keyframe = False
                                break
                    else:
                        curr_features = []
                        for frame_object in self.yolo_v3.final_objects:
                            try:
                                curr_features.append(nhog.HOG(im.fromarray(frame_object)).vector)  
                            except ValueError:
                                curr_features = []
                                is_last_keyframe = False
                                break
                        compare_times = min(len(last_key_features), len(curr_features))
                        similarities = []
                        for i in range(compare_times):
                            similarities.append(nhog.calc_similarity(last_key_features[i], curr_features[i]))
                            # print("similarity of object %d is %f" % (i, similarities[i]))
                            self.message_listbox.insert('end', "[%s]yolo v3 detected %s %d, similarity is %f" % (detected_time, self.yolo_v3.final_classes_names[i], i, similarities[i]))
                            if(similarities[i] < 0.5):
                                self.db.insert(detected_time, self.yolo_v3.final_classes_names[i], \
                                    'yolo v3', self.yolo_v3.final_relative_positions[i][0], self.yolo_v3.final_relative_positions[i][1],\
                                        self.yolo_v3.final_relative_positions[i][2], self.yolo_v3.final_relative_positions[i][3],\
                                            nhog.HOG(im.fromarray(self.yolo_v3.final_objects[i])).vector)
                                self.message_listbox.insert('end', "[%s]yolo v3 store an object tuple into the database" % detected_time)
                        last_key_features = copy.deepcopy(curr_features)
                        cv2image = cv2.cvtColor(self.yolo_v3.img_detection, cv2.COLOR_BGR2RGBA)
                else:
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
                    is_last_keyframe = False
                    last_key_features = []
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image = img)
                self.video_rd.imgtk = imgtk
                self.video_rd.configure(image = imgtk)
                last_frame = frame



    def layout_construct(self):
        '''to build the whole layout and start the main loop. 

        According to the quite disintegrated tkinter documents, 
        first, a 'window' is necessary. Everything will happen here
        and each statement of widget need that 'window' as its first 
        parameter. So for the most part of this function, is to state
        widgets and set their position.

        As for how to show real time video on the 'window', I tried some 
        different ways, but not all worked very well. 
        I initially tried the API in tkinter Label named 'after', to 
        recursively call 'self.show_frame_xx' above to achieve this. But 
        it only worked well when there is one video Label widget. 
        Next I thought about a way to  

        '''
        # main window
        self.window = tk.Tk()
        self.window.title('camera object detection')
        self.window.geometry('2000x1000')
        self.window.resizable(width = True,height = True)

        # sub-windows 
        # left top video sub-window, for original video
        self.video_lt = tk.Label(self.window, bg = 'black')
        self.video_lt.place(x = 5, y = 5, anchor = 'nw')
        # left down video sub-window, for yolo v1
        self.video_ld = tk.Label(self.window, bg = 'black')
        self.video_ld.place(x = 5, y = 500, anchor = 'nw')
        # right top video sub-window, for yolo v2
        self.video_rt = tk.Label(self.window, bg = 'black')
        self.video_rt.place(x = 705, y = 5, anchor = 'nw')
        # right down video sub-window, for yolo v3
        self.video_rd = tk.Label(self.window, bg = 'black')
        self.video_rd.place(x = 705, y = 500, anchor = 'nw')

        # text labels for each video window
        self.text_lt = tk.Label(self.window, text = "original video", bg = 'yellow')
        self.text_lt.place(x = 5, y = 5, anchor = 'nw')

        self.text_ld = tk.Label(self.window, text = "yolo version 1", bg = 'yellow')
        self.text_ld.place(x = 5, y = 500, anchor = 'nw')

        self.text_rt = tk.Label(self.window, text = "yolo version 2", bg = 'yellow')
        self.text_rt.place(x = 705, y = 5, anchor = 'nw')

        self.text_lt = tk.Label(self.window, text = "yolo version 3", bg = 'yellow')
        self.text_lt.place(x = 705, y = 500, anchor = 'nw')

        # buttons
        # trigger for yolo v1
        self.trigger_yolov1 = tk.Button(self.window, text = 'trigger yolo v1', bd = 6, command = self.detection_on_click_1)
        self.trigger_yolov1.place(x = 1625, y = 100, anchor = 'nw')
        # trigger for yolo v2
        self.trigger_yolov2 = tk.Button(self.window, text = 'trigger yolo v2', bd = 6, command = self.detection_on_click_2)
        self.trigger_yolov2.place(x = 1625, y = 150, anchor = 'nw')
        # trigger for yolo v3
        self.trigger_yolov3 = tk.Button(self.window, text = 'trigger yolo v3', bd = 6, command = self.detection_on_click_3)
        self.trigger_yolov3.place(x = 1625, y = 200, anchor = 'nw')
        # switch for original video
        self.switch_video_lt = tk.Button(self.window, text = 'switch lt', bd = 6, command = self.video_on_click_0)
        self.switch_video_lt.place(x = 1735, y = 100, anchor = 'nw')
        # switch for left down video
        self.switch_video_ld = tk.Button(self.window, text = 'switch ld', bd = 6, command = self.video_on_click_1)
        self.switch_video_ld.place(x = 1735, y = 150, anchor = 'nw')
        # switch for right top
        self.switch_video_rt = tk.Button(self.window, text = 'switch rt', bd = 6, command = self.video_on_click_2)
        self.switch_video_rt.place(x = 1735, y = 200, anchor = 'nw')
        # switch for left down video
        self.switch_video_rd = tk.Button(self.window, text = 'switch rd', bd = 6, command = self.video_on_click_3)
        self.switch_video_rd.place(x = 1735, y = 250, anchor = 'nw')

        self.switch_whole = tk.Button(self.window, text = 'switch', bd = 6, command = self.switch_on_click)
        self.switch_whole.place(x = 1625, y = 250, anchor = 'nw')

        # add another listbox and scrollbar to show the messages
        self.scrollbar = tk.Scrollbar(self.window)
        # self.scrollbar.pack(side = tk.RIGHT,fill = tk.Y)
        self.scrollbar.place(x = 1850, y = 400, anchor = 'nw')
        self.message_listbox = tk.Listbox(self.window, yscrollcommand = self.scrollbar.set, height = 30, width = 60)
        self.message_listbox.place(x = 1430, y = 400, anchor = 'nw')
        self.scrollbar.config(command = self.message_listbox.yview)

        # call show frame and main loop functions
        self.cap = cv2.VideoCapture(0)
        Thread(target = self.show_frame_lt, daemon = True).start()
        Thread(target = self.show_frame_ld, daemon = True).start()
        Thread(target = self.show_frame_rt, daemon = True).start()
        Thread(target = self.show_frame_rd, daemon = True).start()
        self.window.mainloop()

    
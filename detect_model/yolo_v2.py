# -*- coding:utf-8 -*-
import random
import colorsys
import tensorflow as tf
import numpy as np
import cv2
import detect_model.configv2 as cfg 


class YOLONetv2():
    def __init__(self):
        # to load the config file into this class 
        self.input_size = cfg.INPUT_SIZE
        self.output_size = cfg.OUTPUT_SIZE
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.anchors = cfg.ANCHORS
        self.num_anchors = len(self.anchors)
        self.weights_file = cfg.WEIGHTS_FILE
        # self.meta_file = cfg.META_FILE
        # self.final_objects = []
        
        self.build_network()
        
        

    def set_image(self,image):
        self.final_objects = []
        self.final_classes_names = []
        self.final_relative_positions = []
        self.image = image
        self.image_shape = self.image.shape[:2]
        self.img_cp = self.preprocess()

    def start(self):
        
        
        # let me make it clear here
        # when using others' trained welled model, first thing to do 
        # is to use the printModel.py to check the tensor name 
        # make sure that your neural network's tensor name is the same as the model
        bboxes,obj_probs,class_probs = self.sess.run(self.output_decoded,feed_dict={self.x:self.img_cp})
        #print(bboxes)
        bboxes,scores,class_max_index = self.postprocess(bboxes,obj_probs,class_probs)


        self.img_detection = self.generate_bounding_box(bboxes,scores,class_max_index)
        self.objects_class_name(class_max_index)

    def conv_layer(self, name, inputs, filters, size, pad_size, stride = 1):
        if(pad_size > 0):
            inputs = tf.pad(inputs,[[0,0], [pad_size, pad_size], [pad_size,pad_size], [0,0]])
        conv = tf.layers.conv2d(inputs, filters = filters, kernel_size = size, strides = stride, padding = 'VALID', use_bias = False, name = name)
        conv = tf.layers.batch_normalization(conv, axis = -1, momentum = 0.9, training = False, name = name + '_bn')
        # this is the same effect as tf.nn.leaky_relu()
        return tf.nn.leaky_relu(conv, alpha = 0.1, name = 'leaky_relu')

    def pooling_layer(self, name, inputs, size, stride):
        return tf.layers.max_pooling2d(inputs, pool_size = size, strides = stride)

    def build_network(self):
        with tf.Graph().as_default() as yolov2_net_graph:
            self.x = tf.placeholder(tf.float32, [1, self.input_size[0], self.input_size[1], 3])
            self.conv_1 = self.conv_layer('conv1', self.x, 32, 3, 1)
            self.pool_2 = self.pooling_layer('pool1', self.conv_1, 2, 2)

            self.conv_3 = self.conv_layer('conv2', self.pool_2, 64, 3, 1)
            self.pool_4 = self.pooling_layer('pool2', self.conv_3, 2, 2)

            self.conv_5 = self.conv_layer('conv3_1', self.pool_4, 128, 3, 1)
            self.conv_6 = self.conv_layer('conv3_2', self.conv_5, 64, 1, 0)
            self.conv_7 = self.conv_layer('conv3_3', self.conv_6, 128, 3, 1)
            self.pool_8 = self.pooling_layer('pool3', self.conv_7, 2, 2)

            self.conv_9 = self.conv_layer('conv4_1', self.pool_8, 256, 3, 1)
            self.conv_10 = self.conv_layer('conv4_2', self.conv_9, 128, 1, 0)
            self.conv_11 = self.conv_layer('conv4_3', self.conv_10, 256, 3, 1)
            self.pool_12 = self.pooling_layer('pool4', self.conv_11, 2, 2)

            self.conv_13 = self.conv_layer('conv5_1', self.pool_12, 512, 3, 1)
            self.conv_14 = self.conv_layer('conv5_2', self.conv_13, 256, 1, 0)
            self.conv_15 = self.conv_layer('conv5_3', self.conv_14, 512, 3, 1)
            self.conv_16 = self.conv_layer('conv5_4', self.conv_15, 256, 1, 0)
            self.conv_17 = self.conv_layer('conv5_5', self.conv_16, 512, 3, 1)
            self.shortcut = self.conv_17
            self.pool_18 = self.pooling_layer('pool5', self.conv_17, 2, 2)

            self.conv_19 = self.conv_layer('conv6_1', self.pool_18, 1024, 3, 1)
            self.conv_20 = self.conv_layer('conv6_2', self.conv_19, 512, 1, 0)
            self.conv_21 = self.conv_layer('conv6_3', self.conv_20, 1024, 3, 1)
            self.conv_22 = self.conv_layer('conv6_4', self.conv_21, 512, 1, 0)
            self.conv_23 = self.conv_layer('conv6_5', self.conv_22, 1024, 3, 1)

            self.conv_24 = self.conv_layer('conv7_1', self.conv_23, 1024, 3, 1)
            self.conv_25 = self.conv_layer('conv7_2', self.conv_24, 1024, 3, 1)

            self.shortcut = self.conv_layer('conv_shortcut', self.shortcut, 64, 1, 0)
            self.shortcut = tf.space_to_depth(self.shortcut, block_size = 2)

            self.conv_26 = tf.concat([self.shortcut, self.conv_25], axis = -1)
            self.conv_27 = self.conv_layer('conv8', self.conv_26, 1024, 3, 1)

            # the last layer is for detection, no BN or activation
            # this also the output layer
            self.model_output = tf.layers.conv2d(self.conv_27, filters = 425, kernel_size = 1, strides = 1, padding = 'VALID', use_bias = True, name = 'conv_dec')
            self.output_decoded = self.interpret_output()
            self.sess = tf.Session(graph = yolov2_net_graph) 
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.weights_file)


    def interpret_output(self):
        # I define all the parameters as the class attributes
        # so there is no need to list here

        # to convert the anchors into tensorflow type
        anchors = tf.constant(self.anchors, dtype = tf.float32)
        H, W = self.input_size[0]//32, self.input_size[1]//32
        # 13 * 13 * num_anchors * (num_class + 5)
        detection_result = tf.reshape(self.model_output, [-1, H*W, self.num_anchors, self.num_classes + 5])

        # to seperate the output tensor into meaningful variables
        xy_offset = tf.nn.sigmoid(detection_result[:,:,:,0:2])
        wh_offset = tf.exp(detection_result[:,:,:,2:4])
        obj_probs = tf.nn.sigmoid(detection_result[:,:,:,4])
        class_probs = tf.nn.sigmoid(detection_result[:,:,:,5:])
        
        # actully this following two variables could be called
        # the left up coordinate of each grid point
        height_index = tf.range(H, dtype=tf.float32)
        width_index = tf.range(W, dtype=tf.float32)
        x_cell, y_cell = tf.meshgrid(height_index, width_index)
        x_cell = tf.reshape(x_cell, [1, -1, 1])
        y_cell = tf.reshape(y_cell, [1, -1, 1])

        bounding_box_x = (x_cell + xy_offset[:,:,:,0]) / W
        bounding_box_y = (y_cell + xy_offset[:,:,:,1]) / H
        bounding_box_w = (anchors[:,0] * wh_offset[:,:,:,0]) / W
        bounding_box_h = (anchors[:,1] * wh_offset[:,:,:,1]) / H

        bounding_boxes = tf.stack([bounding_box_x - bounding_box_w / 2, bounding_box_y - bounding_box_h / 2,
                                 bounding_box_x + bounding_box_w / 2, bounding_box_y + bounding_box_h / 2], axis = 3)
        return bounding_boxes, obj_probs, class_probs                         
        
    def preprocess(self):
        # for safety and easy to debug, we better copy the original image
        image_copy = np.copy(self.image).astype(np.float32)
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        # to resize the image as the size of input_size
        image_resized = cv2.resize(image_rgb, self.input_size)

        image_normalized = image_resized.astype(np.float32) / 225.0

        image_expanded = np.expand_dims(image_normalized, axis = 0)
        
        return image_expanded


    def postprocess(self, bounding_boxes, obj_probs, class_probs):
        # for NMS
        bounding_boxes = np.reshape(bounding_boxes, [-1, 4])
        threshold = 0.5

        # restore the real position in the image
        bounding_boxes[:,0] *= float(self.image_shape[1])
        bounding_boxes[:,1] *= float(self.image_shape[0])
        bounding_boxes[:,2] *= float(self.image_shape[1])
        bounding_boxes[:,3] *= float(self.image_shape[0])
        bounding_boxes = bounding_boxes.astype(np.int32)
        
        # to cut the bounding boxes so that all of them 
        # are within the image bounding 
        min_max = np.array([0, 0, self.image_shape[1] - 1, self.image_shape[0] - 1])
        bounding_boxes[:,0] = np.maximum(bounding_boxes[:,0], min_max[0])
        bounding_boxes[:,1] = np.maximum(bounding_boxes[:,1], min_max[1])
        bounding_boxes[:,2] = np.minimum(bounding_boxes[:,2], min_max[2])
        bounding_boxes[:,3] = np.minimum(bounding_boxes[:,3], min_max[3])
        
        # confidence 
        obj_probs = np.reshape(obj_probs, [-1])
        class_probs = np.reshape(class_probs, [len(obj_probs), -1])
        class_max_index = np.argmax(class_probs, axis = 1)
        
        class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
        confidence = obj_probs * class_probs
        
        # NMS
        # the following two lines used the numpy's feature
        # keep_index is a boolean numpy array
        # class_max_index[keep_index] will only leave the elements
        # whose correspond element in keep_index is True
        keep_index = confidence > threshold
        class_max_index = class_max_index[keep_index]
        confidence = confidence[keep_index]
        bounding_boxes = bounding_boxes[keep_index]
        # sorting and reserve top 400
        index = np.argsort(-confidence)
        class_max_index = class_max_index[index][:400]
        confidence = confidence[index][:400]
        bounding_boxes = bounding_boxes[index][:400]

        keep_bounding_boxes = np.ones(confidence.shape, dtype = np.bool)
        #print(keep_bounding_boxes)
        for i in range(confidence.size - 1):
            if(keep_bounding_boxes[i]):
                overlap = self.iou(bounding_boxes[i], bounding_boxes[(i+1):])
                keep_overlap = np.logical_or(overlap < threshold, class_max_index[(i+1):] != class_max_index[i])
                keep_bounding_boxes[(i+1):] = np.logical_and(keep_bounding_boxes[(i+1):], keep_overlap)
        
        idxes = np.where(keep_bounding_boxes)
        
        return bounding_boxes[idxes], confidence[idxes], class_max_index[idxes]

    def iou(self, bounding_box_1, bounding_box_2):
        bounding_box_1 = np.transpose(bounding_box_1)
        bounding_box_2 = np.transpose(bounding_box_2)
        ymin = np.maximum(bounding_box_1[0], bounding_box_2[0])
        xmin = np.maximum(bounding_box_1[1], bounding_box_2[1])
        ymax = np.minimum(bounding_box_1[2], bounding_box_2[2])
        xmax = np.minimum(bounding_box_1[3], bounding_box_2[3])

        h = np.maximum(ymax - ymin, 0.)
        w = np.maximum(xmax - xmin, 0.)

        # intersection area
        int_area = h * w
        area_1 = (bounding_box_1[2] - bounding_box_1[0]) * (bounding_box_1[3] - bounding_box_1[1])
        area_2 = (bounding_box_2[2] - bounding_box_2[0]) * (bounding_box_2[3] - bounding_box_2[1])
        IoU = int_area / (area_1 + area_2 - int_area)
        return IoU


    def generate_bounding_box(self, bounding_boxes, confidence, class_max_index): # draw detection
        threshold = 0.3
        hsv_tuples = [(x/float(self.num_classes), 1., 1.)  for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        # draw image
        imgcv = np.copy(self.image)
        h, w, _ = imgcv.shape
        for i, box in enumerate(bounding_boxes):
            if confidence[i] < threshold:
                continue
            cls_indx = class_max_index[i]

            # it is quite conveninet to store objects here
            self.final_objects.append(self.image[box[1]:box[3],box[0]:box[2]])
            self.final_relative_positions.append(((box[2]+box[0])/(2*self.image_shape[0]), (box[3]+box[1])/(2*self.image_shape[1]),(box[2]-box[0])/self.image_shape[0] ,(box[3]-box[1])/self.image_shape[1]))# x,y,w,h

            thick = int((h + w) / 300)
            cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),colors[cls_indx], thick)
           

            mess = '%s: %.3f' % (self.classes[cls_indx], confidence[i])
            if box[1] < 20:
                text_loc = (box[0] + 2, box[1] + 15)
            else:
                text_loc = (box[0], box[1] - 10)
            cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (255,255,255), thick//3)
        return imgcv

    def objects_class_name(self, class_index):
        for idx in class_index:
            self.final_classes_names.append(self.classes[idx])
        

    

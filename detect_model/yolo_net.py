import copy 
import numpy as np 
import tensorflow as tf 
import cv2
#import config as cfg
from . import config as cfg

class YOLONet():
    def __init__(self):
        # initialize all parameters
        self.weights_file = cfg.WEIGHTS_FILE
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE 
        self.cell = cfg.CELL 
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.alpha = cfg.ALPHA 
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD

        # final_objects is a list to store the detected objects in an image
        # self.final_objects = []

        self.build_network()
        #self.detect_from_image(image)

    def set_image(self,image):
        self.image = image

        # final_objects is a list to store the detected objects in an image
        # if we reset a new image, we need to reset this list, too 
        self.final_objects = [] 
        self.final_classes_names = []

    # network part
    def build_network(self):
        with tf.Graph().as_default() as yolo_net_graph:
            self.x = tf.placeholder('float32',[None,self.image_size,self.image_size,3])
            self.conv_1 = self.conv_layer(1,self.x,64,7,2)
            self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)
            self.conv_3 = self.conv_layer(3,self.pool_2,192,3,1)
            self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
            self.conv_5 = self.conv_layer(5,self.pool_4,128,1,1)
            self.conv_6 = self.conv_layer(6,self.conv_5,256,3,1)
            self.conv_7 = self.conv_layer(7,self.conv_6,256,1,1)
            self.conv_8 = self.conv_layer(8,self.conv_7,512,3,1)
            self.pool_9 = self.pooling_layer(9,self.conv_8,2,2)
            self.conv_10 = self.conv_layer(10,self.pool_9,256,1,1)
            self.conv_11 = self.conv_layer(11,self.conv_10,512,3,1)
            self.conv_12 = self.conv_layer(12,self.conv_11,256,1,1)
            self.conv_13 = self.conv_layer(13,self.conv_12,512,3,1)
            self.conv_14 = self.conv_layer(14,self.conv_13,256,1,1)
            self.conv_15 = self.conv_layer(15,self.conv_14,512,3,1)
            self.conv_16 = self.conv_layer(16,self.conv_15,256,1,1)
            self.conv_17 = self.conv_layer(17,self.conv_16,512,3,1)
            self.conv_18 = self.conv_layer(18,self.conv_17,512,1,1)
            self.conv_19 = self.conv_layer(19,self.conv_18,1024,3,1)
            self.pool_20 = self.pooling_layer(20,self.conv_19,2,2)
            self.conv_21 = self.conv_layer(21,self.pool_20,512,1,1)
            self.conv_22 = self.conv_layer(22,self.conv_21,1024,3,1)
            self.conv_23 = self.conv_layer(23,self.conv_22,512,1,1)
            self.conv_24 = self.conv_layer(24,self.conv_23,1024,3,1)
            self.conv_25 = self.conv_layer(25,self.conv_24,1024,3,1)
            self.conv_26 = self.conv_layer(26,self.conv_25,1024,3,2)
            self.conv_27 = self.conv_layer(27,self.conv_26,1024,3,1)
            self.conv_28 = self.conv_layer(28,self.conv_27,1024,3,1)
            self.fc_29 = self.fc_layer(29,self.conv_28,512,flat=True,linear=False)
            self.fc_30 = self.fc_layer(30,self.fc_29,4096,flat=False,linear=False)
            #skip dropout_31
            self.fc_32 = self.fc_layer(32,self.fc_30,1470,flat=False,linear=True)
        
            self.sess = tf.Session(graph = yolo_net_graph)
            #tf.initialize_all_variables()
            #THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-02. Instructions for updating: Use tf.global_variables_initializer instead.
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess,self.weights_file)

    def conv_layer(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')#activation

    def pooling_layer(self,idx,inputs,size,stride):
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

    def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
        input_shape = inputs.get_shape().as_list()		
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(inputs,(0,3,1,2))
            inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))	
        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)
        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

    # detection part
    def detect_from_cvmat(self,img):
        self.h_img,self.w_img,_ = img.shape
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray( img_RGB )
        inputs = np.zeros((1,448,448,3),dtype='float32')
        inputs[0] = (img_resized_np/255.0)*2.0-1.0
        in_dict = {self.x: inputs}
        net_output = self.sess.run(self.fc_32,feed_dict=in_dict)
        self.result = self.interpret_output(net_output[0])
        self.generate_bounding_box(img)

    def detect_from_image(self):
        img = self.image
        self.detect_from_cvmat(img)

    def interpret_output(self,output):
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
        
        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
                    probs_filtered[j] = 0.0
        
        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result


    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

    def generate_bounding_box(self,image):
        self.final_with_bounding_box = copy.deepcopy(image)
        
        # the number of objects detected
        self.num_objects = len(self.result)

        for i in range(self.num_objects):
            left_up_x = int(self.result[i][1] - self.result[i][3]/2)
            left_up_y = int(self.result[i][2] - self.result[i][4]/2)
            right_down_x = int(self.result[i][1] + self.result[i][3]/2)
            right_down_y = int(self.result[i][2] + self.result[i][4]/2)
            self.final_objects.append(image[left_up_y:right_down_y,left_up_x:right_down_x])
            self.final_classes_names.append(self.result[i][0])
            cv2.rectangle(self.final_with_bounding_box,(left_up_x,left_up_y),(right_down_x,right_down_y),(0,0,255),2)
            # to add some note above the objects, the second parameter is the text
            # adding the idx 'i' is to see whether the order of objects is relatively unchanged
            # but still hard to say
            cv2.putText(self.final_with_bounding_box,self.result[i][0] + " " + str(i), (left_up_x,left_up_y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)








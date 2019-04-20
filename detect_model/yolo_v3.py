import cv2
import numpy as np 

# using the opencv API to deploy darknet architecture
# this will make the code very easy
# but the .cfg file is very difficult to write
# thanks prior's work

class YOLONetv3():
    def __init__(self):
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (416, 416)
        self.classes_file = "./detect_model/modelv3/coco.names"
        self.classes = None
        with open(self.classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.yolo_v3 = self.build_network()

    def set_image(self, image):
        self.image = image 
        self.final_objects = []
        self.final_classes_names = []
        self.final_relative_positions = []
        blob = cv2.dnn.blobFromImage(self.image, 1/255, self.input_size, [0,0,0], 1, crop = False) 
        self.yolo_v3.setInput(blob)
        
    def build_network(self):
        net = cv2.dnn.readNetFromDarknet("./detect_model/modelv3/yolov3.cfg", "./detect_model/modelv3/yolov3.weights")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    
    def get_output_names(self, network):
        layers_names = network.getLayerNames()
        return [layers_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]

    def generate_bounding_box(self, class_id, confidence, left, top, right, bottom):
        H, W, _ = self.image.shape
        cv2.rectangle(self.imgcv, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % confidence
        if(self.classes):
            label = '%s:%s' % (self.classes[class_id], label)

        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        box = [left, top - round(1.5 * label_size[1]), left + round(1.5 * label_size[0]), top + base_line]
        # store the objects in the list
        self.final_relative_positions.append(((box[2] + box[0])/(2*W), (box[3] + box[1])/(2*H) ,(box[2] - box[0])/W, (box[3] - box[1])/H))# x,y,w,h
        self.final_objects.append(self.image[box[1]:box[3],box[0]:box[2]])
        self.final_classes_names.append(self.classes[class_id])
        cv2.rectangle(self.imgcv, (box[0], box[1]), (box[2], box[3]), (255,255,255), cv2.FILLED)
        cv2.putText(self.imgcv, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        

    def postprocess(self, outputs):
        H, W, _ = self.image.shape
        class_ids = []
        confidences = []
        boxes = []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if(confidence > self.confidence_threshold):
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    width = int(detection[2] * W)
                    height = int(detection[3] * H)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        self.imgcv = np.copy(self.image)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.generate_bounding_box(class_ids[i], confidences[i], left, top, left + width, top + height)

    def start(self):
        '''start the yolo v3 detection

        Start the yolo v3 detection, only can be called after 
        self.set_image()
        '''
        outputs = self.yolo_v3.forward(self.get_output_names(self.yolo_v3))
        self.postprocess(outputs)
        self.img_detection = self.imgcv
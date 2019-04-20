import os


WEIGHTS_FILE = './detect_model/YOLO_small.ckpt'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']


# model hyper parameters

IMAGE_SIZE = 448

CELL = 7

BOXES_PER_CELL = 2

ALPHA = 0.1  #leaky relu parameter

#weights in loss function 
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

# solver parameter

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 45

MAX_ITER = 15000

SUMMARY_ITER = 10

SAVE_ITER = 1000


# test parameter

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
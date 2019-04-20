import os


INPUT_SIZE = (416, 416)

# 32 * 13 = 416
OUTPUT_SIZE = (13,13)

CLASSES = ['person','bicycle','car','motorbike','aeroplane','bus','train',
            'truck','boat','traffic light','fire hydrant','stop sign','parking meter',
            'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
            'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',
            'snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
            'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
            'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
            'donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet',
            'tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
            'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
            'hair drier','toothbrush']

ANCHORS = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

# META_FILE = './modelv2/yolo2_coco.ckpt.meta'
WEIGHTS_FILE = './detect_model/modelv2/yolo2_coco.ckpt'
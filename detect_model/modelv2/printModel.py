import os
from tensorflow.python import pywrap_tensorflow

model_path = 'yolo2_coco.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(model_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ",key)

    
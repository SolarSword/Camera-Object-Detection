# this is a more offcial k-means learning program
# input some command line parameters, you will start the
# k-means training, so make sure the thing you want to train
# is in front of the camera lens
# python k_means_learn.py class_name remark_name
# class_name is the class you want to specify and remark_name is 
# your specify name
import sys
import random
import copy
import pymysql
import cv2
import numpy as np
from PIL import Image as im 
import hog_PIL as nhog
from sample import Sample 
# since YOLO version 2 has the best performance, we use it to 
# collect data
from detect_model.yolo_v2 import YOLONetv2

class_name = str(sys.argv[1])
remark_name = str(sys.argv[2])


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# set up model object
yolo = YOLONetv2()
print("k-means using YOLO v2 collecting data now")
print("make sure the object you want to train is before the camera len")
print("slightly moving your object around to obtain better training data")
print("press 'q' to finish the data collecting")

feature_vectors = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if (not ret):
        break

    yolo.set_image(frame)
    yolo.start()
    cv2.imshow('k-means training processing', yolo.img_detection)
    for i in range(len(yolo.final_objects)):
        if(yolo.final_classes_names[i] == class_name):
            feature_vectors.append(nhog.HOG(im.fromarray(yolo.final_objects[i])).vector) 

     
    if (cv2.waitKey(1) &0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

if(len(feature_vectors) == 0):
    print("Nothing you want detected! Pleas adjust your camera or the object position!")
else:

    samples = []
    for vector in feature_vectors:
        samples.append(Sample(np.array(vector)))

    centroids_number = 3
    centroids = []
    samples_number_around_centroid = []

    for i in range(centroids_number):
        centroids.append(samples[random.randint(0,len(samples)-1)])
        samples_number_around_centroid.append(1)

    candidate_centroid_sum = copy.deepcopy(centroids)
    for i in range(4):
        for sample in samples:
            # compute the distance
            distances = list(map(lambda x: sample.distance(x), centroids))
            # assign the label
            sample.set_label(np.argmin(distances))
            samples_number_around_centroid[sample.label] += 1
            # update the centroid
            candidate_centroid_sum[sample.label] += sample
            candidate_centroid = candidate_centroid_sum[sample.label] / samples_number_around_centroid[sample.label]
            if(candidate_centroid.distance(centroids[sample.label]) > 0.5):
                centroids[sample.label] = copy.deepcopy(candidate_centroid)

    #print(centroids[0].distance(centroids[1]))
    #print(centroids[0].distance(centroids[2]))
    #print(centroids[1].distance(centroids[2]))
    #print((centroids[0].distance(centroids[1])+centroids[0].distance(centroids[2])+centroids[1].distance(centroids[2]))/3)
    c = centroids[0] + centroids[1] + centroids[2]
    c = c / 3

    db = pymysql.connect("localhost", "root", "0305", "objectfeature")
    cursor = db.cursor()
    sql_insert = "insert into centroids(class, name, num_samples, vector)\
                values('%s', '%s', '%d', '%s')" % (class_name, remark_name, len(feature_vectors), c.vector.tolist())
    cursor.execute(sql_insert)
    db.commit()
    db.close()
    cursor.close()            
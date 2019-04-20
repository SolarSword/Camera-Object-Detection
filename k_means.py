# At the very beginning! I must make it clear!
# this is just a test program!!
# but it was very important during my test!
# so I put this program into this project
# NEVER use it after the system started!!!!!  
import copy 
import random
import numpy as np
import matplotlib.pyplot as plt
import pymysql  
from sample import Sample 
import database as db

# first to generate a set of samples 
# choose different random range to generate different clusters
def get_samples():
    samples = []
    results = db.Database().inquire(class_name = 'person')
    for result in results:
        temp = result[0].decode()
        temp = temp[1:][:-1]
        temp = temp.split(',')# now temp is a list, but the elements are string, convert them into int
        temp = list(map(lambda x: int(x), temp))
        samples.append(Sample(np.array(temp)))
    return samples

samples = get_samples()

db = pymysql.connect("localhost","root","0305","objectfeature")
cursor = db.cursor()
sql_fetch = "select num_samples from centroids where name = 'Endong'"
cursor.execute(sql_fetch)
result = cursor.fetchall()
#temp = result[0][0].decode()[1:][:-1].split(',')
#centroid = Sample(np.array(list(map(lambda x: float(x), temp))))

#dis_sum = 0

#for sample in samples:
#    dis_sum += sample.distance(centroid)

#dis_mean = dis_sum / len(samples)
#print(dis_mean)

# the following part is the training part
# once done, never use it, so I comment them out

'''
# centroids initialization
def get_centroids(samples):
    centroids_number = 3
    centroids = []
    samples_number_around_centroid = []

    for i in range(centroids_number):
        centroids.append(samples[random.randint(0,len(samples)-1)])
        samples_number_around_centroid.append(1)
    return centroids, samples_number_around_centroid

centroids, samples_number_around_centroid = get_centroids(samples)   

candidate_centroid_sum = copy.deepcopy(centroids)
# this is actually only once iteration
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

#for centroid in centroids:
#    print(centroid.vector)

print(centroids[0].distance(centroids[1]))
print(centroids[0].distance(centroids[2]))
print(centroids[1].distance(centroids[2]))
print((centroids[0].distance(centroids[1])+centroids[0].distance(centroids[2])+centroids[1].distance(centroids[2]))/3)
c = centroids[0] + centroids[1] + centroids[2]
c = c / 3
number = len(samples)



db = pymysql.connect("localhost","root","0305","objectfeature")
cursor = db.cursor()
sql_insert = "replace into centroids(name, num_samples, vector)\
                values('%s', '%d', '%s')" % ('Endong', number, c.vector.tolist())
cursor.execute(sql_insert)
db.commit()
db.close()
cursor.close()
'''


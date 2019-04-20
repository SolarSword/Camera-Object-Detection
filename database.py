import pymysql
import numpy as np
from sample import Sample
# this the databse structure
# (id, detected_time, class, model, x_relative, y_relative)

class Database():
    def __init__(self):
        self.db = pymysql.connect("localhost","root","0305","objectfeature")
        self.cursor = self.db.cursor()
        # just get the centroids once, for later use
        sql_fetch = "select * from centroids"
        self.cursor.execute(sql_fetch)
        tuples = self.cursor.fetchall()
        self.centroids = {}
        
        for tuple_ in tuples:
            vector = tuple_[4].decode()[1:][:-1].split(',')  
            vector = Sample(np.array(list(map(lambda x: float(x), vector))))
            centroid = (tuple_[2], vector)
            if(tuple_[1] in self.centroids):
                self.centroids[tuple_[1]].append(centroid)
            else:
                self.centroids[tuple_[1]] = [centroid]
            

    def insert(self, detect_time, class_name, model, x, y, w, h, feature):
        if(not (class_name in self.centroids)):
            sql_insert = "insert into records(detected_time, class, model, x_relative, y_relative, width_relative, height_relative, hog_feature) \
                values('%s', '%s', '%s', '%f', '%f', '%f', '%f', '%s')" % \
                    (detect_time, class_name, model, x, y, w, h, feature)
        else:
            centroid_tuples = self.centroids[class_name]
            dis = []
            for centroid in centroid_tuples:
                dis.append(centroid[1].distance(Sample(np.array(feature))))
            if(min(dis) < 1.2):
                sql_insert = "insert into records(detected_time, class, model, x_relative, y_relative, width_relative, height_relative, remark, hog_feature) \
                    values('%s', '%s', '%s', '%f', '%f', '%f', '%f', '%s', '%s')" % \
                        (detect_time, class_name, model, x, y, w, h, centroid_tuples[np.argmin(dis)][0], feature)
            else:
                sql_insert = "insert into records(detected_time, class, model, x_relative, y_relative, width_relative, height_relative, remark, hog_feature) \
                    values('%s', '%s', '%s', '%f', '%f', '%f', '%f', '%s', '%s')" % \
                        (detect_time, class_name, model, x, y, w, h, 'unknown', feature)
        try:
            self.cursor.execute(sql_insert)
            self.db.commit()
        except Exception:
            self.db.rollback()
    
    def inquire(self, class_name, is_feature = True):
        '''To provide an API to access the database 

        Args:
            class_name: to specify the tuples' class you want to access
            is_feature: to specify whether you want to see the hog feature vector
                        this is designed for get the hog vector and label
        '''
        if(is_feature):
            sql_fetch = "select hog_feature from records where class = '%s'" % class_name
        else:
            sql_fetch = "select class, model, x, y, w, h from records where class = '%s'" % class_name
        self.cursor.execute(sql_fetch)
        return self.cursor.fetchall()

    def __del__(self):
        self.db.close()
        self.cursor.close()
# coding=utf-8
import os
from datetime import timedelta, datetime
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time
import mediapipe_utils as mpu


class FacerecResults:
    """
        Attributes:
        score : detection score
        bbox : detection box [x, y, w, h], normalized [0,1] in the squared image
        name : detection keypoints coordinates [x, y], normalized [0,1] in the squared image
        """
    def __init__(self, score=None, bbox=[], conf=None, name=None, xyz=None):
        self.score = score # Palm detection score 
        self.bbox = bbox # Palm detection box [x, y, w, h] normalized
        self.conf = conf # Palm detection keypoints
        self.name = name
        self.xyz = xyz

    def clear(self):
        self.score = None
        self.bbox = []
        self.conf = None
        self.name = None
        self.xyz = None
        
    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


class FaceRecognition:
    def __init__(self, db_path='databases', frameSize=(1920,1080), rec=False) -> None:
        self.b_rec = rec
        self.read_db(db_path)
        self.databases = db_path
        self.record = False
        self.name = 'new_user'
        self.frameSize = frameSize        
        self.fd = None
        self.arc = None
        self.spatial_data = None
        self.results = FacerecResults()
        

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)


    def set_db_path(self, db_path):
        self.databases = db_path
        self.read_db(db_path)


    def start_recording(self, name):
        self.name = name
        self.record = True
        
        
    def stop_recording(self):
        self.record = False
        self.read_db(self.databases)


    def new_recognition(self, results):
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
        if self.record and name[1] == "UNKNOWN":
            self.create_db(results)
        return name
        
        
    def create_db(self, results):
        if self.name is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name wasn't specified")
                self.printed = True
            return
        print('Saving face...')
        try:
            with np.load(f"{self.databases}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{self.databases}/{self.name}", *db_)
        self.adding_new = False
        

    def read_db(self, databases_path):
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]


    def frameNorm(self, frame_size, bbox):
        normVals = np.full(len(bbox), frame_size[1])
        normVals[::2] = frame_size[0]

        ratio = frame_size[0]/frame_size[1]
        bbox2 = (bbox[0]/ratio, bbox[1], bbox[2]/ratio, bbox[3])
        a=(1-(1/ratio))*0.5

        return ((np.array(bbox2)+ [a, 0, a, 0]) * normVals).astype(int)
       

    def getResults(self):
        
        if self.fd is not None:
            self.results.bbox = []
            for i,det in enumerate(self.fd.detections):
                self.results.bbox.append(self.frameNorm(self.frameSize, (det.xmin, det.ymin, det.xmax, det.ymax)))
        else: return None

        if self.b_rec:
            if self.arc is not None:
                features = np.array(self.arc.getFirstLayerFp16())
                self.results.conf, self.results.name = self.new_recognition(features)
            else: return None
        
        
        if self.spatial_data is not None:
            sd = self.spatial_data.getSpatialLocations()[0]
            self.results.xyz = (sd.spatialCoordinates.x,sd.spatialCoordinates.y,sd.spatialCoordinates.z)
        else: return None

        
        return self.results






#!/usr/bin/env python3
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "./oak_utils"))

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String, Float32MultiArray, Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from depthai_ros_msgs.msg import SpatialDetectionArray, SpatialDetection
from vision_msgs.msg import ObjectHypothesis
from geometry_msgs.msg import Point, Pose, PoseArray
import numpy as np
import cv2
import numpy as np
import math
from pipeline_oak_objects import daiPipeline
    

def frameNorm(frame_size, bbox):
    normVals = np.full(len(bbox), frame_size[1])
    normVals[::2] = frame_size[0]

    ratio = frame_size[0]/frame_size[1]
    bbox2 = (bbox[0]/ratio, bbox[1], bbox[2]/ratio, bbox[3])
    a=(1-(1/ratio))*0.5
    return ((np.array(bbox2)+ [a, 0, a, 0]) * normVals).astype(int)


def objDet_msg(oak, detection):
    #print("entering...")
    bbox = frameNorm((oak.img_w,oak.img_h), (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
    ''' SPATIAL DETECTION MSG '''
    msg = SpatialDetection()
    # Results
    msg_result = ObjectHypothesis()
    msg_result.id = detection.label
    msg_result.score = detection.confidence
    msg.results.append(msg_result)
    
    # bbox
    bbox = frameNorm((oak.img_w,oak.img_h), (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
    xSize = bbox[2] - bbox[0];
    ySize = bbox[3] - bbox[1];
    xCenter = bbox[0] + xSize / 2;
    yCenter = bbox[1] + ySize / 2;
    msg.bbox.center.x = xCenter
    msg.bbox.center.y = yCenter
    msg.bbox.size_x = xSize
    msg.bbox.size_y = ySize
    
    # position
    msg.position.x = detection.spatialCoordinates.x/1000
    msg.position.y = detection.spatialCoordinates.y/1000
    msg.position.z = detection.spatialCoordinates.z/1000

    #print(msg)
    return msg


def handDet_msg(hand, b_landmarks=False):
    #print("entering...")
    ''' SPATIAL DETECTION MSG '''
    msg = SpatialDetection()
    # Results
    msg_result = ObjectHypothesis()
    if b_landmarks:
        if hand.label=="left":
            msg_result.id = 1
        else:
            msg_result.id = 2
        msg_result.score = hand.lm_score
    else:
        msg_result.id = 0
        msg_result.score = hand.pd_score
    msg.results.append(msg_result)
    
    # bbox
    bbox = [int(hand.roi.topLeft().x), int(hand.roi.topLeft().y), int(hand.roi.bottomRight().x), int(hand.roi.bottomRight().y)]
    xSize = bbox[2] - bbox[0];
    ySize = bbox[3] - bbox[1];
    xCenter = bbox[0] + xSize / 2;
    yCenter = bbox[1] + ySize / 2;
    msg.bbox.center.x = xCenter
    msg.bbox.center.y = yCenter
    msg.bbox.size_x = xSize
    msg.bbox.size_y = ySize
    
    # position
    msg.position.x = hand.xyz[0]
    msg.position.y = -hand.xyz[1]
    msg.position.z = hand.xyz[2]

    #gesture
    if b_landmarks:
        msg.tracking_id = hand.gesture
        if hand.gesture is None:
            msg.tracking_id = None
    
    #print(msg)
    return msg


class oakPipeline:
    def __init__(self):
        ''' PARAMS '''
        self.node_name = rospy.get_param('~node_name', 'object_det')
        self.oak_dev_id = rospy.get_param('~oak_id', '1944301021A5EE1200')
        self.oak_frame = rospy.get_param('~oak_frame', 'oak')
        self.b_obj_det = rospy.get_param('~obj_det', False)
        self.b_hand_det = rospy.get_param('~hand_det', False)


        ''' SUBSCRIBERS '''
        self.sub_rec = rospy.Subscriber(self.node_name+"/set_obj_det", Bool, self.cb_obj_det)
        self.sub_rec = rospy.Subscriber(self.node_name+"/set_hand_det", Bool, self.cb_hand_det)


        ''' PUBLISHERS '''
        # Depth image
        self.pub_depth = rospy.Publisher(self.node_name+"/depth/image", Image, queue_size=10)
        self.pub_depth_ci = rospy.Publisher(self.node_name+"/depth/camera_info", CameraInfo, queue_size=10)
        # RGB image
        self.pub_rgb = rospy.Publisher(self.node_name+"/rgb/image", Image, queue_size=10)
        self.pub_rgb_ci = rospy.Publisher(self.node_name+"/rgb/camera_info", CameraInfo, queue_size=10)
        # Detections
        self.pub_obj_detections = rospy.Publisher(self.node_name+"/objects/detections", SpatialDetectionArray, queue_size=10)
        self.pub_obj_marker = rospy.Publisher(self.node_name+"/objects/markers", MarkerArray, queue_size=10)
        # Hand
        self.pub_hand_detections = rospy.Publisher(self.node_name+"/hands/detections", SpatialDetectionArray, queue_size=10)
        self.pub_hand_marker = rospy.Publisher(self.node_name+"/hands/markers", MarkerArray, queue_size=10)
        self.pub_hand_pos = rospy.Publisher(self.node_name+"/hands/poses", PoseArray, queue_size=10)
        

        ''' MESSAGES '''
        self.obj_detectionArray = SpatialDetectionArray()
        self.obj_markerArray = MarkerArray()
        self.hand_detectionArray = SpatialDetectionArray()
        self.hand_markerArray = MarkerArray()
        self.hand_poseArray = PoseArray()
        
        
        ''' OAK PIPELINE OBJECT '''
        self.oak = daiPipeline(frameHeight=1000, 
                                obj_det=self.b_obj_det,
                                hand_det=self.b_hand_det,
                                oak_id = self.oak_dev_id)

                                
        ''' OTHER VARIABLES '''
        self.flag_obj_marker = False
        self.flag_hand_marker = False
        self.b_changePipeline = False
        self.bridge = CvBridge()        
        self.user_color = (255,0,0)

        
    def cb_obj_det(self, data):
        self.b_obj_det = data.data
        self.b_changePipeline = True


    def cb_hand_det(self, data):
        self.b_hand_det = data.data
        self.b_changePipeline = True

        
                
    def loop(self):

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.b_changePipeline:
                self.oak.resetPipeline(frameHeight=1000, 
                                        obj_det=self.b_obj_det,
                                        hand_det=self.b_hand_det)
                self.b_changePipeline = False
            
            # Obtain camera frame and hand landmarks
            frame, depth_frame, obj_det_result, hands = self.oak.next_frame()
            
            
            ''' OBJECT DETECTION AND RECOGNITION '''
            if frame is not None and obj_det_result is not None:
                self.obj_detectionArray.detections.clear()
                self.obj_detectionArray.header.stamp = rospy.Time.now()
                self.obj_detectionArray.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                self.obj_markerArray.markers.clear()
                marker_id = 0
                for detection in obj_det_result:
                    self.obj_detectionArray.detections.append(objDet_msg(self.oak, detection))
                    bbox = frameNorm((self.oak.img_w,self.oak.img_h), (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.putText(frame, self.oak.labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.oak.colors[detection.label], 2)
            
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (bbox[0] + 10, bbox[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (bbox[0] + 10, bbox[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (bbox[0] + 10, bbox[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                
                    obj_marker = Marker()
                    obj_marker.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                    obj_marker.id = marker_id
                    obj_marker.type = obj_marker.SPHERE
                    obj_marker.lifetime = rospy.Duration.from_sec(1)
                    #obj_marker.action = obj_marker.ADD
                    obj_marker.scale.x = 0.05
                    obj_marker.scale.y = 0.05
                    obj_marker.scale.z = 0.05
                    obj_marker.color.r = self.oak.colors[detection.label][2]
                    obj_marker.color.g = self.oak.colors[detection.label][1]
                    obj_marker.color.b = self.oak.colors[detection.label][0]                
                    obj_marker.color.a = 1.0
                    obj_marker.pose.orientation.w = 1.0
                    obj_marker.pose.position.x = detection.spatialCoordinates.x/1000
                    obj_marker.pose.position.y = -detection.spatialCoordinates.y/1000
                    obj_marker.pose.position.z = detection.spatialCoordinates.z/1000
                    self.obj_markerArray.markers.append(obj_marker)
                    marker_id = marker_id + 1
                
                self.flag_obj_marker = True
            else:
                self.flag_obj_marker = False
            
            
            ''' HAND LANDMARKS DETECTION '''
            if frame is not None and hands is not None:
                if len(hands) > 0:
                    self.hand_detectionArray.detections.clear()
                    self.hand_detectionArray.header.stamp = rospy.Time.now()
                    self.hand_detectionArray.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                    self.hand_poseArray.poses.clear()
                    self.hand_poseArray.header.stamp = rospy.Time.now()
                    self.hand_poseArray.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                    self.hand_markerArray.markers.clear()
                    marker_id = 0
                    for hand in hands:
                        self.hand_detectionArray.detections.append(handDet_msg(hand))
                        #(hands[0].rect_points)
                        a = np.array(hand.rect_points)
                        a = tuple(list(np.mean(a,axis=0).astype(int)))
                
                        frame = cv2.line(frame, tuple(hand.rect_points[0]), tuple(hand.rect_points[1]), (255, 0, 0), 2)
                        frame = cv2.line(frame, tuple(hand.rect_points[1]), tuple(hand.rect_points[2]), (255, 0, 0), 2)
                        frame = cv2.line(frame, tuple(hand.rect_points[2]), tuple(hand.rect_points[3]), (255, 0, 0), 2)
                        frame = cv2.line(frame, tuple(hand.rect_points[3]), tuple(hand.rect_points[0]), (255, 0, 0), 2)
                        frame = cv2.circle(frame, a, 20, (255, 0, 0), 1)
                    
                        cv2.rectangle(frame, tuple([int(hand.roi.topLeft().x), int(hand.roi.topLeft().y)]), tuple([int(hand.roi.bottomRight().x),int(hand.roi.bottomRight().y)]), (0,255,0), 2)   
                        
                        #print(hand.roi.x)
                        self.hand_screen_pos = [hand.roi.x/self.oak.img_w,hand.roi.y/self.oak.img_h]
                    
                        hand_pos = Pose()
                        hand_pos.position.x = hand.xyz[0]/1000
                        hand_pos.position.y = -hand.xyz[1]/1000
                        hand_pos.position.z = hand.xyz[2]/1000
                        self.hand_poseArray.poses.append(hand_pos)
                        
                        hand_marker = Marker()
                        hand_marker.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                        hand_marker.id = marker_id
                        hand_marker.type = hand_marker.SPHERE
                        hand_marker.lifetime = rospy.Duration.from_sec(1)
                        #hand_marker.action = hand_marker.ADD
                        hand_marker.scale.x = 0.05
                        hand_marker.scale.y = 0.05
                        hand_marker.scale.z = 0.05
                        hand_marker.color.r = 1.0
                        hand_marker.color.g = 1.0
                        hand_marker.color.b = .0                
                        hand_marker.color.a = 1.0
                        hand_marker.pose.orientation.w = 1.0
                        hand_marker.pose.position.x = hand_pos.position.x
                        hand_marker.pose.position.y = hand_pos.position.y
                        hand_marker.pose.position.z = hand_pos.position.z
                        self.hand_markerArray.markers.append(hand_marker)
                        marker_id = marker_id + 1
                    
                    self.flag_hand_marker = True
                    #print(hands[0].xyz/1000)
            else:
                self.flag_hand_marker = False
            
            if depth_frame is not None:
                ''' DEPTH IMAGE PUBLISHER '''
                depth_ci = CameraInfo()
                depth_ci.width=self.oak.img_w
                depth_ci.height=self.oak.img_h
                M, d, R = self.oak.getCalibrationRight((self.oak.img_w,self.oak.img_h))
                depth_ci.K = M.flatten().tolist()
                depth_ci.D = d.tolist()
                depth_ci.R = R.flatten().tolist()
                depth_ci.P = depth_ci.K[0:4]+ [0.0] + depth_ci.K[4:6] + [0.0, 0.0, 0.0, 1.0, 0.0]
            
                depth_frame_msg = self.bridge.cv2_to_imgmsg(depth_frame, "16UC1")
                depth_frame_msg.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                depth_frame_msg.header.stamp = rospy.Time.now()
                depth_frame_msg.is_bigendian = False
                depth_frame_msg.encoding = "16UC1"
                depth_frame_msg.height = self.oak.img_h
                depth_frame_msg.width = self.oak.img_w
            
                self.pub_depth.publish(depth_frame_msg)
                self.pub_depth_ci.publish(depth_ci)
            
            if frame is not None:
                ''' RGB CAMERA PUBLISHER '''            
                frame_ci = CameraInfo()
                frame_ci.width=self.oak.img_w
                frame_ci.height=self.oak.img_h
                M, d, R = self.oak.getCalibrationRGB((self.oak.img_w,self.oak.img_h))
                frame_ci.K = M.flatten().tolist()
                frame_ci.D = d.tolist()
                frame_ci.R = R.flatten().tolist()
                frame_ci.P = frame_ci.K[0:4]+ [0.0] + frame_ci.K[4:6] + [0.0, 0.0, 0.0, 1.0, 0.0]
            
                frame_msg = self.bridge.cv2_to_imgmsg(frame, "8UC3")
                frame_msg.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                frame_msg.header.stamp = rospy.Time.now()
                frame_msg.is_bigendian = False
                frame_msg.encoding = "8UC3"
                frame_msg.height = self.oak.img_h
                frame_msg.width = self.oak.img_w
            
                self.pub_rgb.publish(frame_msg)
                self.pub_rgb_ci.publish(frame_ci)
            
            
            ''' MARKERS PUBLISHERS '''
            if self.flag_obj_marker:
                self.pub_obj_marker.publish(self.obj_markerArray)
                self.pub_obj_detections.publish(self.obj_detectionArray)
                
            if self.flag_hand_marker:
                self.pub_hand_marker.publish(self.hand_markerArray)
                self.pub_hand_detections.publish(self.hand_detectionArray)
                self.pub_hand_pos.publish(self.hand_poseArray)
            
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('oak_scene')
    oak_pipeline = oakPipeline()
    oak_pipeline.loop()
    #rospy.spin()

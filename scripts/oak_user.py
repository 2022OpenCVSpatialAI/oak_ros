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
import numpy as np
import cv2
import numpy as np
import math
from pipeline_oak_user import daiPipeline
from HandTracker import draw_hand_landmarks



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
            msg.tracking_id = "None"
    
    #print(msg)
    return msg
    

class oakPipeline:
    def __init__(self):
        ''' PARAMS '''
        self.node_name = rospy.get_param('~node_name', 'object_det')
        self.oak_dev_id = rospy.get_param('~oak_id', '18443010A12BBF1200')
        self.oak_frame = rospy.get_param('~oak_frame', 'oak')
        self.b_face_det = rospy.get_param('~face_det', False)
        self.b_face_rec = rospy.get_param('~face_rec', False)
        self.b_hand_det = rospy.get_param('~hand_det', False)
        self.b_hand_landmarks = rospy.get_param('~hand_landmarks', False)
        
        
        ''' SUBSCRIBERS '''
        self.sub_rec = rospy.Subscriber(self.node_name+"/set_face_det", Bool, self.cb_face_det)
        self.sub_rec = rospy.Subscriber(self.node_name+"/set_face_rec", Bool, self.cb_face_rec)
        self.sub_rec = rospy.Subscriber(self.node_name+"/set_hand_det", Bool, self.cb_hand_det)
        self.sub_rec = rospy.Subscriber(self.node_name+"/set_hand_landmarks", Bool, self.cb_hand_landmarks)

        
        ''' PUBLISHERS '''
        # Depth image
        self.pub_depth = rospy.Publisher(self.node_name+"/depth/image", Image, queue_size=10)
        self.pub_depth_ci = rospy.Publisher(self.node_name+"/depth/camera_info", CameraInfo, queue_size=10)
        # RGB image
        self.pub_rgb = rospy.Publisher(self.node_name+"/rgb/image", Image, queue_size=10)
        self.pub_rgb_ci = rospy.Publisher(self.node_name+"/rgb/camera_info", CameraInfo, queue_size=10)
        # Hand
        self.pub_hand_screen_pos = rospy.Publisher(self.node_name+"/hands/hand_screen_pos", Float32MultiArray, queue_size=1)
        self.pub_gesture = rospy.Publisher(self.node_name+"/hands/gesture", String, queue_size=10)     
        self.pub_hand_marker = rospy.Publisher(self.node_name+"/hands/markers", Marker, queue_size=10)
        self.pub_hand_markerArray = rospy.Publisher(self.node_name+"/hands/markersArray", MarkerArray, queue_size=10)
        self.pub_hand_detections = rospy.Publisher(self.node_name+"/hands/detections", SpatialDetectionArray, queue_size=10)
        # User
        self.pub_head_marker = rospy.Publisher(self.node_name+"/face/markers", Marker, queue_size=10)
        self.pub_username = rospy.Publisher(self.node_name+"/face/name", String, queue_size=10)
        self.pub_userdist = rospy.Publisher(self.node_name+"/face/dist", Float32, queue_size=10)     
                        
        
        ''' MESSAGES '''
        self.hand_detectionArray = SpatialDetectionArray()
        self.hand_markerArray = MarkerArray()
        self.hand_marker = Marker()
        self.head_marker = Marker()


        ''' OAK PIPELINE OBJECT '''
        self.oak = daiPipeline(frameHeight=1000, 
                                face_det=self.b_face_det,
                                face_rec=self.b_face_rec, 
                                hand_det=self.b_hand_det,
                                hand_landmarks=self.b_hand_landmarks,
                                oak_id = self.oak_dev_id)
                                
        
        ''' OTHER VARIABLES '''
        self.flag_head_marker = False
        self.flag_hand_marker = False
        self.b_changePipeline = False

        self.bridge = CvBridge()
                
        self.user = 'UNKNOWN'
        self.user_color = (255,0,0)
        self.hand_gesture = ''
        self.hand_screen_pos = []
        self.user_dist = 1000;
        
        
    
    def cb_face_det(self, data):
        self.b_face_det = data.data
        if data.data is False:
            self.b_face_rec = False
        
        self.b_changePipeline = True


    def cb_face_rec(self, data):
        self.b_face_rec = data.data
        if data.data:
            self.b_face_det = True
        
        self.b_changePipeline = True


    def cb_hand_det(self, data):
        self.b_hand_det = data.data
        if data.data is False:
            self.b_hand_landmarks = False
        
        self.b_changePipeline = True


    def cb_hand_landmarks(self, data):
        self.b_hand_landmarks = data.data
        if data.data:
            self.b_hand_det = True
        
        self.b_changePipeline = True
        
                
    def loop(self):

        rate = rospy.Rate(30) # 10hz
        while not rospy.is_shutdown():
            if self.b_changePipeline:
                self.oak.resetPipeline(frameHeight=1000, 
                                        face_det=self.b_face_det,
                                        face_rec=self.b_face_rec, 
                                        hand_det=self.b_hand_det,
                                        hand_landmarks=self.b_hand_landmarks)
                self.user_color = (255,0,0)
                self.b_changePipeline = False
            
            # Obtain camera frame and hand landmarks
            frame, depth_frame, facerec_result, hands = self.oak.next_frame()
            
            
            ''' FACE DETECTION AND RECOGNITION '''
            if frame is not None and facerec_result is not None:
                if self.b_face_rec:
                    self.user = facerec_result.name
                    if self.user == 'UNKNOWN':
                        self.user_color = (0,0,255)
                    elif self.user is None:
                        self.user_color = (255,0,0)
                    else:
                        self.user_color = (0,255,0)
                    
                self.user_dist = facerec_result.xyz[2]/1000
                
                self.head_marker.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                self.head_marker.lifetime = rospy.Duration.from_sec(1)
                self.head_marker.type = self.head_marker.SPHERE
                self.head_marker.action = self.head_marker.ADD
                self.head_marker.scale.x = 0.05
                self.head_marker.scale.y = 0.05
                self.head_marker.scale.z = 0.05
                self.head_marker.color.r = self.user_color[2]
                self.head_marker.color.g = self.user_color[1]
                self.head_marker.color.b = self.user_color[0]                
                self.head_marker.color.a = 1.0
                self.head_marker.pose.orientation.w = 1.0
                self.head_marker.pose.position.x = facerec_result.xyz[0]/1000
                self.head_marker.pose.position.y = -facerec_result.xyz[1]/1000
                self.head_marker.pose.position.z = facerec_result.xyz[2]/1000
                self.flag_head_marker = True
            
                if len(facerec_result.bbox):
                    cv2.rectangle(frame, (facerec_result.bbox[0][0], facerec_result.bbox[0][1]), (facerec_result.bbox[0][2], facerec_result.bbox[0][3]), self.user_color, 2)
            else:
                self.flag_head_marker = False
                # ~ rate.sleep()
                # ~ continue
            
            
            ''' HAND LANDMARKS DETECTION '''
            if frame is not None and hands is not None:
                self.hand_detectionArray.detections.clear()
                self.hand_detectionArray.header.stamp = rospy.Time.now()
                self.hand_detectionArray.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                self.hand_markerArray.markers.clear()
                marker_id = 0
                if len(hands) > 0:
                    for hand in hands:
                        #print(hands[0].rect_points)
                        self.hand_detectionArray.detections.append(handDet_msg(hand,self.b_hand_landmarks))
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
                    
                        hand_color = (1.0,1.0,1.0)
                        if self.b_hand_landmarks:
                            if self.oak.handTrack.process_landmarks:
                                frame = draw_hand_landmarks(frame,hands, False, True)
                                self.hand_gesture = hands[0].gesture
                                if hand.label == "left":
                                    hand_color = (0.0,1.0,1.0)
                                elif hand.label == "right":
                                    hand_color = (1.0,1.0,0.0)
                        
                        hand_marker = Marker()
                        hand_marker.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                        hand_marker.id = marker_id
                        hand_marker.lifetime = rospy.Duration.from_sec(1)
                        hand_marker.type = hand_marker.SPHERE
                        hand_marker.action = hand_marker.ADD
                        hand_marker.scale.x = 0.05
                        hand_marker.scale.y = 0.05
                        hand_marker.scale.z = 0.05
                        hand_marker.color.r = hand_color[0]
                        hand_marker.color.g = hand_color[1]
                        hand_marker.color.b = hand_color[2]         
                        hand_marker.color.a = 1.0
                        hand_marker.pose.orientation.w = 1.0
                        hand_marker.pose.position.x = hand.xyz[0]/1000
                        hand_marker.pose.position.y = -hand.xyz[1]/1000
                        hand_marker.pose.position.z = hand.xyz[2]/1000                    
                        self.hand_markerArray.markers.append(hand_marker)
                        marker_id = marker_id + 1
                    
                    # ~ self.hand_marker.header.frame_id = self.oak_frame+"_right_camera_optical_frame"
                    # ~ self.hand_marker.lifetime = rospy.Duration.from_sec(1)
                    # ~ self.hand_marker.type = self.hand_marker.SPHERE
                    # ~ self.hand_marker.action = self.hand_marker.ADD
                    # ~ self.hand_marker.scale.x = 0.05
                    # ~ self.hand_marker.scale.y = 0.05
                    # ~ self.hand_marker.scale.z = 0.05
                    # ~ self.hand_marker.color.r = 1.0
                    # ~ self.hand_marker.color.g = 1.0
                    # ~ self.hand_marker.color.b = .0                
                    # ~ self.hand_marker.color.a = 1.0
                    # ~ self.hand_marker.pose.orientation.w = 1.0
                    # ~ self.hand_marker.pose.position.x = hands[0].xyz[0]/1000
                    # ~ self.hand_marker.pose.position.y = -hands[0].xyz[1]/1000
                    # ~ self.hand_marker.pose.position.z = hands[0].xyz[2]/1000
                    self.flag_hand_marker = True
                    #print(hands[0].xyz/1000)
            else:
                self.flag_hand_marker = False
            
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
            if self.flag_head_marker:
                self.pub_head_marker.publish(self.head_marker)
                self.pub_userdist.publish(self.user_dist)
                if self.b_face_rec and self.user is not None:
                    self.pub_username.publish(self.user)
                
            if self.flag_hand_marker:
                self.pub_hand_marker.publish(self.hand_marker)
                self.pub_hand_markerArray.publish(self.hand_markerArray)
                msg = Float32MultiArray()
                msg.data = self.hand_screen_pos
                self.pub_hand_screen_pos.publish(msg)
                self.pub_hand_detections.publish(self.hand_detectionArray)
                if self.b_hand_landmarks:
                    self.pub_gesture.publish(self.hand_gesture)           
                    
            
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('hand_test')
    oak_pipeline = oakPipeline()
    oak_pipeline.loop()
    #rospy.spin()

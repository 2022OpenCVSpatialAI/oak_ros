# Code obtained from https://github.com/geaxgx/depthai_hand_tracker

import numpy as np
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS, now


SCRIPT_DIR = Path(__file__).resolve().parent

FINGER_COLOR = [
    (128, 128, 128),
    (80, 190, 168),
    (234, 187, 105),
    (175, 119, 212),
    (81, 110, 221),
]

JOINT_COLOR = [(0, 0, 0), (125, 255, 79), (255, 102, 0), (181, 70, 255), (13, 63, 255)]

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1)#.flatten()
    
def draw_hand_landmarks(img, hands, zoom_mode, single_handed):
    list_connections = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    for hand in hands:
        lm_xy = []
        for landmark in hand.landmarks:
            lm_xy.append([int(landmark[0]), int(landmark[1])])
        palm_line = [np.array([lm_xy[point] for point in [0, 5, 9, 13, 17, 0]])]
        cv2.polylines(img, palm_line, False, (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(len(list_connections)):
            finger = list_connections[i]
            line = [np.array([lm_xy[point] for point in finger])]
            cv2.polylines(img, line, False, FINGER_COLOR[i], 2, cv2.LINE_AA)
            for point in finger:
                pt = lm_xy[point]
                cv2.circle(img, (pt[0], pt[1]), 3, JOINT_COLOR[i], -1)
        if single_handed:
            if zoom_mode:
                cv2.line(img, (lm_xy[4][0], lm_xy[4][1]), (lm_xy[8][0], lm_xy[8][1]), (0, 255, 0), 2, cv2.LINE_AA)
    return img

class HandTracker:
    def __init__(self, input_src=None,
                pd_score_thresh=0.65, pd_nms_thresh=0.3,
                use_lm=True,
                lm_score_thresh=0.5,
                internal_fps=23,
                internal_frame_height=720,
                stats=False,
                trace=False
                ):

        
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_thresh = lm_score_thresh
        self.internal_fps = internal_fps     
        self.stats = stats
        self.trace = trace
        
        self.process_landmarks = True
        
        self.lm_input_length = 224


        # Note that here (in Host mode), specifying "rgb_laconic" has no effect
        # Color camera frames are systematically transferred to the host
        self.input_type = "rgb" # OAK* internal color camera
        self.internal_fps = internal_fps 
        print(f"Internal camera FPS set to: {self.internal_fps}")
        self.resolution = (1920, 1080)

        print("Sensor resolution:", self.resolution)

        self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps
        
        width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
        self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.pad_h = (self.img_w - self.img_h) // 2
        self.pad_w = 0
        self.frame_size = self.img_w
        self.crop_w = 0

        print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")

        # Create SSD anchors 
        self.anchors = mpu.generate_handtracker_anchors()
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")


        self.fps = FPS()

        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.nb_spatial_requests = 0
        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0
        self.glob_spatial_rtrip_time = 0

        self.use_previous_landmarks = False

        self.hand_from_landmarks = {"left": None, "right": None}


    def recognize_gesture(self, r):           
        # Finger states
        # state: -1=unknown, 0=close, 1=open
        d_3_5 = mpu.distance(r.norm_landmarks[3], r.norm_landmarks[5])
        d_2_3 = mpu.distance(r.norm_landmarks[2], r.norm_landmarks[3])
        angle0 = mpu.angle(r.norm_landmarks[0], r.norm_landmarks[1], r.norm_landmarks[2])
        angle1 = mpu.angle(r.norm_landmarks[1], r.norm_landmarks[2], r.norm_landmarks[3])
        angle2 = mpu.angle(r.norm_landmarks[2], r.norm_landmarks[3], r.norm_landmarks[4])
        r.thumb_angle = angle0+angle1+angle2
        if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2: 
            r.thumb_state = 1
        else:
            r.thumb_state = 0

        if r.norm_landmarks[8][1] < r.norm_landmarks[7][1] < r.norm_landmarks[6][1]:
            r.index_state = 1
        elif r.norm_landmarks[6][1] < r.norm_landmarks[8][1]:
            r.index_state = 0
        else:
            r.index_state = -1

        if r.norm_landmarks[12][1] < r.norm_landmarks[11][1] < r.norm_landmarks[10][1]:
            r.middle_state = 1
        elif r.norm_landmarks[10][1] < r.norm_landmarks[12][1]:
            r.middle_state = 0
        else:
            r.middle_state = -1

        if r.norm_landmarks[16][1] < r.norm_landmarks[15][1] < r.norm_landmarks[14][1]:
            r.ring_state = 1
        elif r.norm_landmarks[14][1] < r.norm_landmarks[16][1]:
            r.ring_state = 0
        else:
            r.ring_state = -1

        if r.norm_landmarks[20][1] < r.norm_landmarks[19][1] < r.norm_landmarks[18][1]:
            r.little_state = 1
        elif r.norm_landmarks[18][1] < r.norm_landmarks[20][1]:
            r.little_state = 0
        else:
            r.little_state = -1

        # Gesture
        if r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
            r.gesture = "FIVE"
        elif r.thumb_state == 0 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "FIST"
        elif r.thumb_state == 1 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "ZOOM" 
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "PEACE"
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "ONE"
        elif r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "ZOOM"
        elif r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "THREE"
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
            r.gesture = "FOUR"
        else:
            r.gesture = None
   
    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.hands = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=False)
        # Non maximum suppression (not needed if solo)
        self.hands = mpu.non_max_suppression(self.hands, self.pd_nms_thresh)
        if self.use_lm:
            mpu.detections_to_rect(self.hands)
            mpu.rect_transformation(self.hands, self.img_w, self.img_h)

    def lm_postprocess(self, hand, inference):
        hand.lm_score = inference.getLayerFp16("Identity_1")[0]  
        if hand.lm_score > self.lm_score_thresh:  
            hand.handedness = inference.getLayerFp16("Identity_2")[0]
            lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add")).reshape(-1,3)
            # hand.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
            hand.norm_landmarks = lm_raw / self.lm_input_length
            # hand.norm_landmarks[:,2] /= 0.4

            # Now calculate hand.landmarks = the landmarks in the image coordinate system (in pixel)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in hand.rect_points[1:]], dtype=np.float32) # hand.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(hand.norm_landmarks[:,:2], axis=0)
            lm_z = hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4
            hand.landmarks = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
            hand.landmarks = np.concatenate((hand.landmarks, lm_z), axis=1)
            self.recognize_gesture(hand)

    def spatial_loc_roi_from_palm_center(self, hand):
        half_size = int(hand.pd_box[2] * self.frame_size / 2)
        zone_size = max(half_size//2, 8)
        rect_center = dai.Point2f(int(hand.pd_box[0]*self.frame_size) + half_size - zone_size//2 + self.crop_w, int(hand.pd_box[1]*self.frame_size) + half_size - zone_size//2 - self.pad_h)
        rect_size = dai.Size2f(zone_size, zone_size)
        return dai.Rect(rect_center, rect_size)
        
        
    def setLandmarks(self, flag):
        self.process_landmarks = flag
        
        
    def process(self, video_frame, pd_inference, q_lm_in, q_lm_out, q_hand_sp_data_out, q_hand_sp_cfg_in):
        bag = {}
        bag["body"] = None
        self.fps.update()

        # Get palm detection
        if not self.use_previous_landmarks:
            self.pd_postprocess(pd_inference)
            self.nb_pd_inferences += 1  
        else:
            self.hands = [self.hand_from_landmarks["right"], self.hand_from_landmarks["left"]]

        if self.pad_h:
            square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
        else:
            square_frame = video_frame
        
        for h in self.hands:
            if h.pd_box is not None:
                h.roi = self.spatial_loc_roi_from_palm_center(h)
                cfg_info = dai.SpatialLocationCalculatorConfig()
                data_cfg_xyz = dai.SpatialLocationCalculatorConfigData()
                data_cfg_xyz.depthThresholds.lowerThreshold = 100
                data_cfg_xyz.depthThresholds.upperThreshold = 10000
                data_cfg_xyz.roi = h.roi
                cfg_info.addROI(data_cfg_xyz)
                q_hand_sp_cfg_in.send(cfg_info)
                sd = q_hand_sp_data_out.get().getSpatialLocations()[0]
                h.xyz = (sd.spatialCoordinates.x,sd.spatialCoordinates.y,sd.spatialCoordinates.z)
        
        if self.process_landmarks:
            # Hand landmarks, send requests
            if self.use_lm and len(self.hands)>0:
                for i,h in enumerate(self.hands):
                    img_hand = mpu.warp_rect_img(h.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
                    #cv2.imshow("color2", img_hand)
                    #key_cmd = cv2.waitKey(1)
                    nn_data = dai.ImgFrame()
                    nn_data.setWidth(self.lm_input_length)
                    nn_data.setHeight(self.lm_input_length)
                    nn_data.setData(
                        to_planar(
                            img_hand,
                            (
                                self.lm_input_length,
                                self.lm_input_length,
                            ),
                        )
                    )
                    q_lm_in.send(nn_data)
                    if i == 0: lm_rtrip_time = now() # We measure only for the first hand
                for i,h in enumerate(self.hands):
                    inference = None
                    inference = q_lm_out.tryGet()
                    if inference is None:
                        h.lm_score=0
                        continue
                    if i == 0: self.glob_lm_rtrip_time += now() - lm_rtrip_time
                    self.lm_postprocess(h, inference)
                    self.nb_lm_inferences += 1
                bag["lm_inference"] = len(self.hands)
                temp_hands = [ h for h in self.hands if h.lm_score > self.lm_score_thresh]
                #print(self.hands[0].lm_score, self.lm_score_thresh)
                if len(temp_hands) > 0:
                    self.hands = [temp_hands[0]]
                    for hand in temp_hands[1:]:
                        if abs(hand.handedness - self.hands[0].handedness) > 0.4:
                            self.hands.append(hand)
                else:
                    #self.hands = []
                    return None
            

            
                for hand in self.hands:
                    # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
                    if self.pad_h > 1000:
                        hand.landmarks[:,1] -= self.pad_h
                        for i in range(len(hand.rect_points)):
                            hand.rect_points[i][1] -= self.pad_h
                    if self.pad_w > 0:
                        hand.landmarks[:,0] -= self.pad_w
                        for i in range(len(hand.rect_points)):
                            hand.rect_points[i][0] -= self.pad_w

                    # Set the hand label
                    hand.label = "right" if hand.handedness > 0.5 else "left"       
        
            
        return self.hands


    def exit(self):
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nb_frames()})")
            if self.body_pre_focusing:
                print(f"# body pose estimation inferences received : {self.nb_bpf_inferences}")
            print(f"# palm detection inferences received       : {self.nb_pd_inferences}")
            if self.use_lm: print(f"# hand landmark inferences received        : {self.nb_lm_inferences}")
            if self.input_type != "rgb":
                if self.body_pre_focusing:
                    print(f"Body pose estimation round trip      : {self.glob_bpf_rtrip_time/self.nb_bpf_inferences*1000:.1f} ms")
                print(f"Palm detection round trip            : {self.glob_pd_rtrip_time/self.nb_pd_inferences*1000:.1f} ms")
                if self.use_lm and self.nb_lm_inferences:
                    print(f"Hand landmark round trip             : {self.glob_lm_rtrip_time/self.nb_lm_inferences*1000:.1f} ms")
            if self.xyz:
                print(f"Spatial location requests round trip : {self.glob_spatial_rtrip_time/self.nb_anchors*1000:.1f} ms")           

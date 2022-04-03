# coding=utf-8
import os
from pathlib import Path
import blobconverter
import cv2
import depthai as dai
import numpy as np
from string import Template
import mediapipe_utils as mpu
from facerecognition import FaceRecognition
from HandTracker import HandTracker, draw_hand_landmarks

SCRIPT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = str(SCRIPT_DIR / "./databases")
FACE_DETECTION_MODEL = str(SCRIPT_DIR / "./models/face-detection-retail-0004_openvino_2021.3_2shave.blob")
FACE_RECOGNITION_MODEL = str(SCRIPT_DIR / "./models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob")
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "./models/palm_detection_sh4.blob")
LANDMARK_MODEL = str(SCRIPT_DIR / "./models/hand_landmark_sparse_sh4.blob")
SCRIPT_FILE = str(SCRIPT_DIR / "script.py")


class daiPipeline:
    def __init__(self, 
                frameHeight = 1152, 
                face_det=True, 
                face_rec=False, 
                hand_det=False, 
                hand_landmarks=False,
                oak_id = "18443010A12BBF1200") -> None:
		
        ''' PIPELINE INFORMATION '''
        self.b_face_det = face_det
        self.b_face_rec = face_rec
        self.b_hand_det = hand_det
        self.b_hand_landmarks = hand_landmarks
        self.oak_id = oak_id

        ''' IMAGE INFORMATION '''
        self.resolution = (1920,1080)
        self.frameHeight = frameHeight
        
        
        ''' DEFINE AND START PIPELINE '''
        self.pipeline_running = False
        self.reset_pipeline = False
        self.setPipeline()
                
        
    def setPipeline(self):
        ''' IMAGE INFORMATION '''
        width, self.scale_nd = mpu.find_isp_scale_params(self.frameHeight, self.resolution, is_height=False)
        self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.ratio = self.img_w/self.img_h
        #print((self.img_h, self.img_w, self.pad_h, self.pad_w))
        
        ''' DEFINE AND START PIPELINE '''
        dev_state, dev_info = dai.Device.getDeviceByMxId(self.oak_id)
        pipeline = self.create_pipeline_init()
        pipeline = self.pipeline_cameras(pipeline)
        if self.b_face_det:
            pipeline = self.pipeline_face_det(pipeline, self.b_face_rec)
            
        if self.b_hand_det:
            pipeline = self.pipeline_hand_det(pipeline, self.b_hand_landmarks)
        
        if dev_state:
            self.device = dai.Device(pipeline, dev_info)
        else:
            self.device = dai.Device(pipeline)
        usb_speed = self.device.getUsbSpeed()
        self.calibData = self.device.readCalibration()
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")
        
        ''' DEFINE DATA QUEUES '''
        # Cameras queues
        self.frameQ = self.device.getOutputQueue("frame", 4, False)
        self.depthQ = self.device.getOutputQueue("depth_out", 4, False)
        # Face det queues
        if self.b_face_det:
            self.face_det_manipQ = self.device.getOutputQueue("face_det_manip", 4, False)
            self.face_detQ = self.device.getOutputQueue("face_det", 4, False)
            self.spatial_dataQ = self.device.getOutputQueue(name="spatial_data_out", maxSize=4, blocking=False)
            if self.b_face_rec:
                self.arcQ = self.device.getOutputQueue("arc_out", 4, False)
            
        if self.b_hand_det:
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            self.q_hand_sp_data_out = self.device.getOutputQueue(name="hand_sp_data_out")
            self.q_hand_sp_cfg_in = self.device.getInputQueue(name="hand_sp_cfg_in")
            if self.b_hand_landmarks:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")
            
            
        ''' DEFINE OTHER VARIABLES '''
        if self.b_face_det:
            self.faceRec = FaceRecognition(db_path=DATABASE_PATH ,frameSize=(self.img_w, self.img_h), rec=self.b_face_rec)
            self.facerec_result = None
            
        if self.b_hand_det:
            self.handTrack = HandTracker(internal_frame_height=self.img_h)
            self.handTrack.setLandmarks(False)
            self.hands = None
            if self.b_hand_landmarks:
                self.handTrack.setLandmarks(True)
            
        self.pipeline_running = True
    
    def resetPipeline(self, face_det=True, face_rec=False, hand_det=False, hand_landmarks=False, frameHeight=1000):
        self.pipeline_running = False
        
        self.device.close()
        ''' PIPELINE INFORMATION '''
        self.b_face_det = face_det
        self.b_face_rec = face_rec
        self.b_hand_det = hand_det
        self.b_hand_landmarks = hand_landmarks

        ''' IMAGE INFORMATION '''
        self.frameHeight = frameHeight
        
        self.setPipeline()
        
    def getCalibrationLeft(self, resolution):
        M = np.array(self.calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution[0], resolution[1]))
        d = np.array(self.calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
        R = np.array(self.calibData.getStereoLeftRectificationRotation())
        return M, d, R
        
    def getCalibrationRight(self, resolution):
        M = np.array(self.calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resolution[0], resolution[1]))
        d = np.array(self.calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
        R = np.array(self.calibData.getStereoRightRectificationRotation())
        return M, d, R
        
    def getCalibrationRGB(self, resolution):
        M = np.array(self.calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, resolution[0], resolution[1]))
        d = np.array(self.calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        R = np.array(self.calibData.getStereoRightRectificationRotation())
        return M, d, R
        
    def create_pipeline_init(self):
        print("Creating face pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_3)
        openvino_version = '2021.3'
        
        return pipeline
        
    def pipeline_cameras(self, pipeline):
        ''' CAMERA RGB NODE '''
        self.cam = pipeline.create(dai.node.ColorCamera)
        self.cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        self.cam.setVideoSize(self.img_w, self.img_h)
        self.cam.setPreviewSize(self.img_w, self.img_h)
        #self.cam.setPreviewSize(1152, 648)
        #self.cam.setVideoSize(1152, 648)        
        self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam.setInterleaved(False)
        self.cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        
        ''' MONOCAMERA NODE '''
        mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_480_P
        # Left camera
        self.left = pipeline.createMonoCamera()
        self.left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.left.setResolution(mono_resolution)
        # Right camera
        self.right = pipeline.createMonoCamera()
        self.right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.right.setResolution(mono_resolution)
        
        ''' STEREO NODE '''
        # Stereo node
        self.stereo = pipeline.createStereoDepth()
        self.stereo.setConfidenceThreshold(230)
        # LR-check is required for depth alignment
        self.stereo.setLeftRightCheck(True)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        #self.stereo.setExtendedDisparity(True)
        self.stereo.setSubpixel(True)  # subpixel True brings latency
        self.stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        
        ''' SPATIAL_LOCATION_CALCULATOR NODE '''
        self.spatial_location_calculator = pipeline.createSpatialLocationCalculator()
        self.spatial_location_calculator.setWaitForConfigInput(True)
        self.spatial_location_calculator.inputDepth.setBlocking(False)
        self.spatial_location_calculator.inputDepth.setQueueSize(1)
        
        ''' SCRIPTS NODE '''
        self.script = pipeline.create(dai.node.Script)
        self.script.setProcessor(dai.ProcessorType.LEON_CSS)
        self.script.setScript(self.build_manager_script())
        # ~ with open(SCRIPT_FILE, "r") as f:
            # ~ self.script.setScript(f.read())
        
        ''' PIPELINE CONNECTIONS '''
        # Link Left camera -> stereo node
        self.left.out.link(self.stereo.left)
        # Link right camera -> stereo node
        self.right.out.link(self.stereo.right) 
        # Link stereo depth -> spatial calculator node
        self.stereo.depth.link(self.spatial_location_calculator.inputDepth)
        
        if self.b_face_det or self.b_hand_det:
            # Link script ['depth_cfg'] -> spatial calculator node
            self.script.outputs['depth_cfg'].link(self.spatial_location_calculator.inputConfig)
            # Link spatial calculator node -> script ['spatial_data']
            self.spatial_location_calculator.out.link(self.script.inputs['spatial_data'])
        
        ''' XLINKS NODES '''
        # XLinkOut RGB camera image
        cam_out = pipeline.create(dai.node.XLinkOut)
        cam_out.setStreamName('frame')
        self.cam.preview.link(cam_out.input)
        
        # XLinkOut depth image
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName('depth_out')
        self.stereo.depth.link(depth_out.input)
        
        return pipeline
        
    def build_manager_script(self):
        # Read the template
        with open(SCRIPT_FILE, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _FACE_DET = "True" if self.b_face_det else "False",
                    _FACE_REC = "True" if self.b_face_rec else "False",
                    _HAND_DET = "True" if self.b_hand_det else "False",
        )

        # ~ # For debugging
        # ~ with open("tmp_code.py", "w") as file:
            # ~ file.write(code)

        return code

    def pipeline_face_det(self, pipeline, b_rec = False):
        ''' IMAGE MANIP NODE '''
        # ImageManip that will crop the frame before sending it to the Face detection NN node
        face_det_manip = pipeline.create(dai.node.ImageManip)
        face_det_manip.initialConfig.setResize(300, 300)
        face_det_manip.initialConfig.setKeepAspectRatio(True)
        face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
        
        ''' NN NODE '''
        # Creating Face Detection Neural Network
        face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        face_det_nn.setConfidenceThreshold(0.5)
        face_det_nn.setBlobPath(FACE_DETECTION_MODEL)
        
        ''' PIPELINE CONNECTIONS '''
        # Link Camera Preview -> Face_Det ImageManip
        self.cam.preview.link(face_det_manip.inputImage)
        # Link Face_Det ImageManip -> Script ['preview']
        face_det_manip.out.link(self.script.inputs['preview'])
        # Link Face_Det ImageManip -> Face detection NN node
        face_det_manip.out.link(face_det_nn.input)        
        # Link Face detection NN node -> Script['face_det_in']
        face_det_nn.out.link(self.script.inputs['face_det_in'])
        
        
        ''' XLINKS NODES '''
        # XLinkOut face_det NN input image
        face_det_manip_out = pipeline.create(dai.node.XLinkOut)
        face_det_manip_out.setStreamName('face_det_manip')
        face_det_manip.out.link(face_det_manip_out.input)

        # XLinkOut face_det NN output image
        face_det_out = pipeline.create(dai.node.XLinkOut)
        face_det_out.setStreamName('face_det')
        face_det_nn.out.link(face_det_out.input)
            
        # XLinkOut spatial location information
        spatial_data_out = pipeline.createXLinkOut()
        spatial_data_out.setStreamName("spatial_data_out")
        spatial_data_out.input.setQueueSize(1)
        spatial_data_out.input.setBlocking(False)
        self.script.outputs['spatial_data_out'].link(spatial_data_out.input)
        
        
        if b_rec:
            ''' IMAGE MANIP NODE '''
            # Manip to preprocess for face_rec NN
            face_rec_manip = pipeline.create(dai.node.ImageManip)
            face_rec_manip.initialConfig.setResize(112, 112)
        
            ''' NN NODE '''
            # Creating Face Recognition Neural Network
            face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
            face_rec_nn.setBlobPath(FACE_RECOGNITION_MODEL)
            
            ''' PIPELINE CONNECTIONS '''
            # Link Script['manip_cfg'] -> Face_Rec ImageManip
            self.script.outputs['manip_cfg'].link(face_rec_manip.inputConfig)
            # Link Script['manip_img'] -> Face_Rec ImageManip
            self.script.outputs['manip_img'].link(face_rec_manip.inputImage)
            # Link Face_Rec ImageManip -> Face recognition NN node
            face_rec_manip.out.link(face_rec_nn.input)
            
            # XLinkOut face_det NN output information
            arc_out = pipeline.create(dai.node.XLinkOut)
            arc_out.setStreamName('arc_out')
            face_rec_nn.out.link(arc_out.input)
    
        return pipeline
        
    def pipeline_hand_det(self, pipeline, b_landmark=False):
        ''' HAND DETECTION NODES '''
        manip = pipeline.createImageManip()
        manip.setMaxOutputFrameSize(128*128*3)
        manip.initialConfig.setResizeThumbnail(128, 128)
        manip.setWaitForConfigInput(False)
        manip.inputImage.setQueueSize(1)
        manip.inputImage.setBlocking(False)
 
        # Define palm detection model
        # Creating Palm Detection Neural Network
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(PALM_DETECTION_MODEL)
        pd_nn.input.setQueueSize(1)
        pd_nn.input.setBlocking(False)
        

        ''' PIPELINE CONNECTIONS '''
        self.cam.preview.link(manip.inputImage)
        manip.out.link(pd_nn.input)
        
        ''' XLINKS NODES '''
        # Manipulation input
        manip_cfg_in = pipeline.createXLinkIn()
        manip_cfg_in.setStreamName("manip_cfg")
        manip_cfg_in.out.link(manip.inputConfig)
        # Palm detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
        
        # XLinkOut spatial location information
        hand_sp_data_out = pipeline.createXLinkOut()
        hand_sp_data_out.setStreamName("hand_sp_data_out")
        hand_sp_data_out.input.setQueueSize(1)
        hand_sp_data_out.input.setBlocking(False)
        self.script.outputs['hand_sp_data_out'].link(hand_sp_data_out.input)
    
        hand_sp_cfg_in = pipeline.createXLinkIn()
        hand_sp_cfg_in.setStreamName("hand_sp_cfg_in")
        hand_sp_cfg_in.out.link(self.script.inputs['hand_sp_cfg_in'])
        
        
        if self.b_hand_landmarks:
            # Define hand landmark model
            print("Creating Hand Landmark Neural Network...")          
            lm_nn = pipeline.createNeuralNetwork()
            lm_nn.setBlobPath(LANDMARK_MODEL)
            # Hand landmark input
            lm_in = pipeline.createXLinkIn()
            lm_in.setStreamName("lm_in")
            lm_in.out.link(lm_nn.input)
            # Hand landmark output
            lm_out = pipeline.createXLinkOut()
            lm_out.setStreamName("lm_out")
            lm_nn.out.link(lm_out.input)

        return pipeline
    

    def next_frame(self):
        
        frame = None
        depth_frame = None
        facerec_result = None
        inference = None
        hands = None

        if self.pipeline_running:
            frameIn = self.frameQ.get()
            if frameIn is not None:
                frame = frameIn.getCvFrame()
            
            depthIn = self.depthQ.get()
            if depthIn is not None:
                depth_frame = depthIn.getCvFrame()
                
            if self.b_face_det:
                self.faceRec.fd = self.face_detQ.tryGet()
                if self.b_face_rec:
                    self.faceRec.arc = self.arcQ.tryGet()
                else:
                    self.faceRec.arc = None

                self.faceRec.spatial_data = self.spatial_dataQ.tryGet()
                facerec_result = self.faceRec.getResults()
                #if facerec_result is not None:
                #    self.facerec_result = facerec_result
                self.facerec_result = facerec_result
            # ~ else:
                # ~ self.facerec_result = None

            
            if self.b_hand_det:
                inference = self.q_pd_out.tryGet()
                if inference is not None and frame is not None:
                    if self.b_hand_landmarks:
                        hands = self.handTrack.process(frame, inference, self.q_lm_in, self.q_lm_out, self.q_hand_sp_data_out, self.q_hand_sp_cfg_in)
                    else:
                        hands = self.handTrack.process(frame, inference, None, None, self.q_hand_sp_data_out, self.q_hand_sp_cfg_in)
                    self.hands = hands

        return frame, depth_frame, self.facerec_result, self.hands


    def exit(self):
        self.device.close()



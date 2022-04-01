# coding=utf-8
import os
from pathlib import Path
import blobconverter
import cv2
import depthai as dai
import numpy as np
from string import Template
import mediapipe_utils as mpu
from HandTracker import HandTracker
import json

SCRIPT_DIR = Path(__file__).resolve().parent
NN_CONFIG_FILE = str(SCRIPT_DIR / "./models/yolo-tiny.json")
OBJECT_DETECTION_MODEL = str(SCRIPT_DIR / "./models/yolo_v4_tiny_sh4.blob")
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "./models/palm_detection_sh4.blob")
SCRIPT_FILE = str(SCRIPT_DIR / "script.py")

COLORS=[(255,0,0), (0,0,255), (0,255,0)]

class daiPipeline:
    def __init__(self, 
                frameHeight = 1152, 
                obj_det=True, 
                hand_det=False,
                oak_id = "1944301021A5EE1200") -> None:
		
        ''' PIPELINE INFORMATION '''
        self.b_obj_det = obj_det
        self.b_hand_det = hand_det
        self.oak_id = oak_id

        ''' IMAGE INFORMATION '''
        self.resolution = (1920,1080)
        self.frameHeight = frameHeight
        
        
        ''' DEFINE AND START PIPELINE '''
        self.pipeline_running = False
        self.reset_pipeline = False
        self.labels = None
        self.colors = []
        self.setPipeline()
                
        
    def setPipeline(self):
        ''' IMAGE INFORMATION '''
        width, self.scale_nd = mpu.find_isp_scale_params(self.frameHeight, self.resolution, is_height=False)
        self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.ratio = self.img_w/self.img_h
        #print((self.img_h, self.img_w, self.pad_h, self.pad_w))
        
        ''' DEFINE AND START PIPELINE '''
        #dev_state, dev_info = dai.Device.getDeviceByMxId('1944301021A5EE1200')
        #dev_state, dev_info = dai.Device.getDeviceByMxId('18443010A12BBF1200')
        dev_state, dev_info = dai.Device.getDeviceByMxId(self.oak_id)
        
        pipeline = self.create_pipeline_init()
        pipeline = self.pipeline_cameras(pipeline)
        if self.b_obj_det:
            pipeline = self.pipeline_obj_det(pipeline)
            
        if self.b_hand_det:
            pipeline = self.pipeline_hand_det(pipeline)
        
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
        # Obj det queues
        if self.b_obj_det:
            self.obj_detQ = self.device.getOutputQueue("obj_det", 4, False)
            
        if self.b_hand_det:
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            self.q_hand_sp_data_out = self.device.getOutputQueue(name="hand_sp_data_out")
            self.q_hand_sp_cfg_in = self.device.getInputQueue(name="hand_sp_cfg_in")
            
            
        ''' DEFINE OTHER VARIABLES '''
        self.obj_det_result = None
            
        if self.b_hand_det:
            self.handTrack = HandTracker(internal_frame_height=self.img_h)
            self.handTrack.setLandmarks(False)
            self.hands = None
            
        self.pipeline_running = True
    
    def resetPipeline(self, obj_det=True, hand_det=False, frameHeight=1000):
        self.pipeline_running = False
        
        self.device.close()
        ''' PIPELINE INFORMATION '''
        self.b_obj_det = obj_det
        self.b_hand_det = hand_det

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
        #self.stereo.setSubpixel(True)  # subpixel True brings latency
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
        
        ''' PIPELINE CONNECTIONS '''
        # Link Left camera -> stereo node
        self.left.out.link(self.stereo.left)
        # Link right camera -> stereo node
        self.right.out.link(self.stereo.right) 
        # Link stereo depth -> spatial calculator node
        self.stereo.depth.link(self.spatial_location_calculator.inputDepth)
        
        if self.b_hand_det:
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
                    _FACE_DET = "False",
                    _FACE_REC = "False",
                    _HAND_DET = "True" if self.b_hand_det else "False",
        )

        # ~ # For debugging
        # ~ with open("tmp_code.py", "w") as file:
            # ~ file.write(code)

        return code

    def pipeline_obj_det(self, pipeline):
        ''' YOLO PARAMETERS FROM FILE '''
        configPath = Path(NN_CONFIG_FILE)
        with configPath.open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        # parse labels
        self.colors = []
        nnMappings = config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})
        print(self.labels)
        for i in range(len(self.labels)):
            self.colors.append(COLORS[i])

        # parse input shape
        if "input_size" in nnConfig:
            W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        classes = metadata.get("classes", {})
        coordinates = metadata.get("coordinates", {})
        anchors = metadata.get("anchors", {})
        anchorMasks = metadata.get("anchor_masks", {})
        iouThreshold = metadata.get("iou_threshold", {})
        confidenceThreshold = metadata.get("confidence_threshold", {})
        
        ''' IMAGE MANIP NODE '''
        # ImageManip that will crop the frame before sending it to the Face detection NN node
        obj_det_manip = pipeline.create(dai.node.ImageManip)
        obj_det_manip.initialConfig.setResize(W, H)
        #obj_det_manip.initialConfig.setKeepAspectRatio(True)
        obj_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
        
        ''' NN NODE '''
        # Creating Face Detection Neural Network
        detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        detectionNetwork.setConfidenceThreshold(confidenceThreshold)
        detectionNetwork.setNumClasses(classes)
        detectionNetwork.setCoordinateSize(coordinates)
        detectionNetwork.setAnchors(anchors)
        detectionNetwork.setAnchorMasks(anchorMasks)
        detectionNetwork.setIouThreshold(iouThreshold)
        detectionNetwork.setBlobPath(OBJECT_DETECTION_MODEL)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)
        
        ''' PIPELINE CONNECTIONS '''
        self.cam.preview.link(obj_det_manip.inputImage)
        obj_det_manip.out.link(detectionNetwork.input)
        self.stereo.depth.link(detectionNetwork.inputDepth)
        
        ''' XLINKS NODES '''
        # XLinkOut face_det NN output image
        obj_det_out = pipeline.create(dai.node.XLinkOut)
        obj_det_out.setStreamName('obj_det')
        detectionNetwork.out.link(obj_det_out.input)
    
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
        
        return pipeline
    

    def next_frame(self):
        
        frame = None
        depth_frame = None
        obj_det_result = None
        inference = None
        hands = None

        if self.pipeline_running:
            frameIn = self.frameQ.get()
            if frameIn is not None:
                frame = frameIn.getCvFrame()
            
            depthIn = self.depthQ.get()
            if depthIn is not None:
                depth_frame = depthIn.getCvFrame()
                
            if self.b_obj_det:
                inDet = self.obj_detQ.tryGet()
                if inDet is not None:
                    self.obj_det_result = inDet.detections                
                # ~ else:
                    # ~ self.obj_det_result = obj_det_result
            
            
            if self.b_hand_det:
                inference = self.q_pd_out.tryGet()
                if inference is not None and frame is not None:
                    hands = self.handTrack.process(frame, inference, None, None, self.q_hand_sp_data_out, self.q_hand_sp_cfg_in)
                    self.hands = hands

        return frame, depth_frame, self.obj_det_result, self.hands


    def exit(self):
        self.device.close()



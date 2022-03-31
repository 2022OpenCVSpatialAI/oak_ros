import time
bboxes = [] # List of face BBs
l = [] # List of images


def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.0
    if bb.ymin < 0: bb.ymin = 0.0
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb
    
    
b_face_det = ${_FACE_DET}
b_face_rec = ${_FACE_REC}
b_hand_det = ${_HAND_DET}
    
while True:
    time.sleep(0.001)
    
    ''' FACE DETECTION'''
    if b_face_det:
        
        preview = node.io['preview'].tryGet()
        if preview is None:
            continue

        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
            min_z = [0, 10000]
            xyz_out = None
            for i,det in enumerate(face_dets.detections):
                a=(1-(1/1.77))*0.5

                cfg_xyz = SpatialLocationCalculatorConfig()
                data_cfg_xyz = SpatialLocationCalculatorConfigData()
                data_cfg_xyz.depthThresholds.lowerThreshold = 100
                data_cfg_xyz.depthThresholds.upperThreshold = 10000
                data_cfg_xyz.roi = Rect(Point2f(a+det.xmin/1.77, det.ymin), Point2f(a+det.xmax/1.77, det.ymax))
                if data_cfg_xyz.roi.topLeft().x <0: data_cfg_xyz.roi.topLeft().x=0
                if data_cfg_xyz.roi.topLeft().y <0: data_cfg_xyz.roi.topLeft().y=0
                if data_cfg_xyz.roi.bottomRight().x <0: data_cfg_xyz.roi.bottomRight().x=0
                if data_cfg_xyz.roi.bottomRight().y <0: data_cfg_xyz.roi.bottomRight().y=0
            
                if data_cfg_xyz.roi.topLeft().x >1: data_cfg_xyz.roi.topLeft().x=0.999
                if data_cfg_xyz.roi.topLeft().y >1: data_cfg_xyz.roi.topLeft().y=0.999
                if data_cfg_xyz.roi.bottomRight().x >1: data_cfg_xyz.roi.bottomRight().x=0.999
                if data_cfg_xyz.roi.bottomRight().y >1: data_cfg_xyz.roi.bottomRight().y=0.999
                                    
                cfg_xyz.addROI(data_cfg_xyz)
                node.io['depth_cfg'].send(cfg_xyz)
            
                info_xyz = node.io['spatial_data'].get()
                data_xyz = info_xyz.getSpatialLocations()
                #node.warn(str([i, data_xyz[0].spatialCoordinates.z]))
                if data_xyz[0].spatialCoordinates.z < min_z[1]:
                    min_z = [i, data_xyz[0].spatialCoordinates.z]
                    xyz_out = info_xyz
                
            if xyz_out is not None:
        
                #node.warn('Out: '+str(min_z))
            
                if b_face_rec:
                    det_near = face_dets.detections[min_z[0]]
                    cfg = ImageManipConfig()
                    correct_bb(det_near)
                    cfg.setCropRect(det_near.xmin, det_near.ymin, det_near.xmax, det_near.ymax)
                    #node.warn(str((det.xmin, det.ymin, det.xmax, det.ymax)))
                    cfg.setResize(112, 112)
                    cfg.setKeepAspectRatio(True)
            
                    node.io['manip_cfg'].send(cfg)
                    node.io['manip_img'].send(preview)
            
                node.io['spatial_data_out'].send(xyz_out)
            
            
    ''' HAND DETECTION '''
    if b_hand_det:
        hand_sp_cfg = node.io['hand_sp_cfg_in'].tryGet()
        if hand_sp_cfg is not None:
            node.io['depth_cfg'].send(hand_sp_cfg)
            hand_xyz = node.io['spatial_data'].get()
            node.io['hand_sp_data_out'].send(hand_xyz)

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np

# Creating Detector Class
class Detector:
    def __init__(self, model_type = "OD"): #object detection
        self.cfg = get_cfg()
        self.model_type = model_type

        #Load model config and pretrained model
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_config("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        elif model_type == "IS": #Instance Segmentation
            self.cfg.merge_from_file(model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type == "KP": #Keypoint Detection
            self.cfg.merge_from_file(model_zoo.get_config("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_config("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        elif model_type == "LVIS": #LVIS Segmentation
            self.cfg.merge_from_file(model_zoo.get_config("LVISv0.5-InstanceSegmentation/mask_rcnn_x_101_32x8d_Fpn_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_config("LVISv0.5-InstanceSegmentation/mask_rcnn_x_101_32x8d_Fpn_1x.yaml")

        elif model_type == "KP": #Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_config("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7        
        self.cfg.MODEL.DEVICE = "cpu"  #cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)


    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            predictions = self.predictor(image)


            viz = Visualizer(image[:,:,::-1], Metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE_BW) #SEGMENTATION  or IMAGE

            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)


    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened()==False):
            print("Error")
            return

        (sucess, image) = cap.read()

        while success:
            if self.model_type != "PS":
                predictions = self.predictor(image)


                viz = Visualizer(image[:,:,::-1], Metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode = ColorMode.IMAGE_BW) #SEGMENTATION  or IMAGE

                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            else:
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            cv2.imshow("Result", output.get_image()[:,:,::-1])


            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
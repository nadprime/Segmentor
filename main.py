from Detector import *

detector = Detector(model_type= "IS") #IS or OD or KP
detector = Detector(model_type= "OD")
detector = Detector(model_type= "PS")
detector = Detector(model_type= "LVIS")
detector = Detector(model_type= "KP")

detector.onImage("images/1.jpg")

detector.onVideo("video1.mp4")
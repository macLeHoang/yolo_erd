from ultralytics import YOLO
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModelErd


if __name__ == "__main__":
    

    
    model = YOLO("last.pt")
    model("ngoc_nhay.mp4", save=True, imgsz=(640, 960), iou=0.4, agnostic_nms=True, conf=0.25)

    # ckpt = torch.load("best (1).pt")
    # print(ckpt["ema"].state_dict().keys())

    # model = YOLO("best (1).pt")
    # state_dict = model.model.state_dict()
    # print(state_dict['model.22.dfl.conv.weight'].size())

    # model_t = YOLO("yolov8m.pt")
    # state_dict_t = model_t.model.state_dict()
    # print(state_dict_t['model.22.dfl.conv.weight'].size())

    # 'model.22.cv3.2.1.bn.running_mean', 
    # 'model.22.cv3.2.1.bn.running_var', 
    # 'model.22.cv3.2.1.bn.num_batches_tracked', 
    # 'model.22.cv3.2.2.weight', 
    # 'model.22.cv3.2.2.bias', 
    # 'model.22.dfl.conv.weight'

    # a = DetectionModelErd()

    # print(model.model.model[-1])

    # ckpt = torch.load("yolov8s.pt")
    # model = YOLO("best (1).pt")

    # model.model.load(ckpt)

    # print(ckpt["model"].state_dict()["model.22.cv3.2.1.bn.running_mean"].shape==
    #       model.model.state_dict()["model.22.cv3.2.1.bn.running_mean"].shape)



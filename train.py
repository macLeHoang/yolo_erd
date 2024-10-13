from ultralytics import YOLO


if __name__ == "__main__":
    model_t = YOLO("yolov8m.pt")
    model = YOLO("yolov8m.pt")
    model.train(data="coco8.yaml", epochs=50, batch=16, imgsz=640, Continuous=model_t.model, 
                multi_scale=False, device=0,
                mosaic=1.0, 
                cls=0.75, kl_gain=0.75, cls_disstil_gain=1.0)

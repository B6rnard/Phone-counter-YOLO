from ultralytics import YOLO

# Indlæs YOLO-model
model = YOLO("yolo11n.pt") 

# Start træning
model.train(data="datasets/data.yaml", epochs=100, batch=2, imgsz=360)

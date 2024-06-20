import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Device is: ", device)

model = YOLO("yolov8n.pt")

model.to(device)

model.train(data="D:\myenv\Dice_Game\data.yaml", epochs=50)

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = YOLO("/content/drive/MyDrive/table-tennis-dataset/yolov8_output/custom_yolo_training/weights/best.pt")

video_path = "//content/drive/MyDrive/table-tennis-dataset/video/video_03.mp4"
cap = cv2.VideoCapture(video_path)

points_x, points_y = [], []
table_width = 152.5
table_length = 274
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.3)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            mapped_x = (cx / frame_width) * table_width
            mapped_y = (cy / frame_height) * table_length
            points_x.append(mapped_x)
            points_y.append(mapped_y)

cap.release()

plt.figure(figsize=(10, 6))
sns.kdeplot(x=points_x, y=points_y, fill=True, cmap="Reds", bw_adjust=0.5, levels=100)
plt.title("Heatmap Analysis of Table Tennis Scoring Ball Landing Points")
plt.xlabel("Table Width (cm)")
plt.ylabel("Table Length (cm)")
plt.gca().invert_yaxis()
plt.show()

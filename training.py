from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics

from ultralytics import YOLO

# 訓練資料集設定檔路徑
dataset_path = "/content/drive/MyDrive/table-tennis-dataset/data.yaml"

# 自訂儲存目錄（包含模型權重 best.pt）
output_dir = "/content/drive/MyDrive/table-tennis-dataset/yolov8_output"

# 建立模型並訓練
model = YOLO("yolov8n.pt")
model.train(
    data=dataset_path,
    epochs=50,
    imgsz=640,
    batch=16,
    project=output_dir,
    name="custom_yolo_training",
    exist_ok=True
)

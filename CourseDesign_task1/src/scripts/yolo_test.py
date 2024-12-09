import cv2
from ultralytics import YOLO


# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法访问摄像头")
    exit()

# Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("pt/yolo11s.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
# results = model.val()

while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧，退出")
        break

    # 使用YOLOv8模型进行检测
    results = model.predict(source=frame,show=True)

    # 解析检测结果并绘制
    # frame = results.plot()  # 自动在图像上绘制检测框和标签

    # 显示带有检测框的实时视频流
    # cv2.imshow("YOLOv8 Detection", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()

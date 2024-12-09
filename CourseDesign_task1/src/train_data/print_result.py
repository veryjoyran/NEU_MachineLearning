from ultralytics import YOLO
import time

model=YOLO("/home/joyran/NEU_MachineLearning/runs/detect/train8/weights/best.pt")

result=model.predict(source="/home/joyran/NEU_MachineLearning/CourseDesign_task1/src/data/test/0091.jpg",show=True)

time.sleep(5)

print(result)


# 遍历结果列表，提取中心点坐标
all_centers = []

for res in result:  # 遍历每个结果对象
    if res.boxes is not None:  # 检查是否有检测框
        boxes = res.boxes.xyxy  # 获取检测框的坐标 (x1, y1, x2, y2)
        for box in boxes:  # 遍历每个检测框
            x1, y1, x2, y2 = box  # 解包坐标
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            all_centers.append((center_x.item(), center_y.item()))  # 添加中心点坐标

print("所有检测框的中心点坐标:", all_centers)


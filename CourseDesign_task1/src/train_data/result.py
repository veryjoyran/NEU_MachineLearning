import csv
from ultralytics import YOLO
import time
import os

model = YOLO("/CourseDesign_task1/src/best.pt")

image_dir = "/home/joyran/NEU_MachineLearning/CourseDesign_task1/src/data/test/"

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort(key=lambda x: int(x.split('.')[0][1:]))

# CSV 文件路径
csv_file = 'detection_results.csv'

# 创建并打开 CSV 文件写入数据
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # 写入 CSV 表头
    writer.writerow(['ImageID', 'value'])

    # 遍历所有图片
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        # 对每张图片进行检测
        result = model.predict(source=image_path, show=False)

        # time.sleep(5)

        # 遍历每个结果对象，提取检测框的中心点坐标
        for res in result:
            if res.boxes is not None:  # 检查是否有检测框
                boxes = res.boxes.xyxy  # 获取检测框的坐标 (x1, y1, x2, y2)
                for box in boxes:  # 遍历每个检测框
                    x1, y1, x2, y2 = box  # 解包坐标
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # 写入 CSV 文件，按照格式 ImageID_Fovea_X 和 ImageID_Fovea_Y
                    image_id = os.path.splitext(image_file)[0]
                    image_id=int(image_id)
                    writer.writerow([f"{image_id}_Fovea_X", center_x.item()])
                    writer.writerow([f"{image_id}_Fovea_Y", center_y.item()])

print(f"检测结果已保存到: {csv_file}")

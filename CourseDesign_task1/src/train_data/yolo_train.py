from ultralytics import YOLO

# Load a model
model = YOLO("/home/joyran/NEU_MachineLearning/CourseDesign_task1/src/scripts/pt/yolo11s.pt")  #

# Train the model
results = model.train(data="/home/joyran/NEU_MachineLearning/CourseDesign_task1/src/train_data/dataset_test/dataset_test.yaml", epochs=1000, imgsz=640)
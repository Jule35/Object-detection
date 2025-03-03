from ultralytics import YOLO

model = YOLO('yolov8l')  # Load model

result = model.predict('test5_s.mp4',save = True, conf=0.2, iou=0.9)


print(result[0])
for box in result[0].boxes:
      if box.conf > 0.2 :
        print(box)  


import cv2
import os
from ultralytics import YOLO
from functions import function_system

yolov8_path = '../model/yolov8_model/yolov8_model_2_augmentations.pt'
yolov8_model = YOLO(yolov8_path)

video_path = '../data/testing/video_testing_3.mp4'
cap = cv2.VideoCapture(video_path)

count_image = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        detect_object = yolov8_model.predict(frame)
        annotated_frame = function_system.plot_bboxes(frame, detect_object[0].boxes.data, conf=0.5)

        directory = '../data/testing/frame/skenario_3/output_video_3'
        filename = 'results_images_' + str((count_image + 1)) + '.jpg'

        cv2.imwrite(os.path.join(directory, filename), annotated_frame[0])

        count_image += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()

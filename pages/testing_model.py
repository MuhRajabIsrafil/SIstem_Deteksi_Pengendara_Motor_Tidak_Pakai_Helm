import cv2
import os
from ultralytics import YOLO
from functions import function_system

# Mengatur path ke model YOLOv8 yang telah dilatih
yolov8_path = '../model/yolov8_model/yolov8_model_2_augmentations(skenario3).pt'
yolov8_model = YOLO(yolov8_path)

# Menentukan path video yang akan diproses
video_path = '../data/training/videos/video_5.mp4'

# Membuka video untuk pemrosesan frame-by-frame
cap = cv2.VideoCapture(video_path)

# Inisialisasi penghitung gambar yang telah diproses
count_image = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Mendeteksi objek pada frame menggunakan YOLOv8
        detect_object = yolov8_model.predict(frame)
        annotated_frame = function_system.plot_bboxes(frame, detect_object[0].boxes.data, conf=0.5)

        # Menentukan direktori untuk menyimpan hasil gambar
        directory = '../data/training/videos/training_5'
        filename = 'results_images_' + str((count_image + 1)) + '.jpg'

        # Menyimpan gambar hasil deteksi ke file
        cv2.imwrite(os.path.join(directory, filename), annotated_frame[0])

        count_image += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()

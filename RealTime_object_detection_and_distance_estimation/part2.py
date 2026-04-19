
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("C:/Users/HP/Documents/opencv_projects/video1.mp4")  

KNOWN_WIDTH = 0.07   # meters
FOCAL_LENGTH = 240  

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("End of video")
        break
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            pixel_width = x2 - x1
            
            if pixel_width > 0:
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
                
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                if confidence < 0.6:
                    continue
                label = f"{class_name} {confidence*100:.2f}% | {distance:.2f}m"
                

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,255,0), 1)
                print(f"Pixel Width: {pixel_width}, Distance: {distance:.2f} m")
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Distance Estimation", frame)
    delay=int(1000/30)  # 30 FPS
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
# doing the distance estimation using the formula: Distance = (Known Width * Focal Length) / Pixel Width
# all of this is happening with the laptop camera, so the focal length is fixed, and the known width is also fixed, so we can calculate the distance of the object from the camera using the pixel width of the object in the image.

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

KNOWN_WIDTH = 0.07   # meters
FOCAL_LENGTH = 240  # adjust later

while True:
    ret, frame = cap.read()
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            pixel_width = x2 - x1
            
            if pixel_width > 0:
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

                label = f"{distance:.2f} m"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                print(f"Pixel Width: {pixel_width}, Distance: {distance:.2f} m")

    cv2.imshow("Distance Estimation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
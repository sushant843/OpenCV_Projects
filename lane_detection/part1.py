import cv2
import numpy as np  


img = cv2.imread(r"C:\Users\HP\Documents\opencv_projects\testimg4.jpg")
if img is None:
    print("Error: Image not found.")
else: 
    img = cv2.resize(img, (1280, 720)) 
    cv2.imshow("Road Image", img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150,apertureSize=3)
    height, width = canny.shape
    mask = np.zeros_like(canny)*255
    
    roi_corners = np.array([[(200, height), (width - 100, height), (width // 2, 250)]], np.int32)
    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(canny, mask)
    lines = cv2.HoughLinesP(canny, 10, np.pi/180, 50, np.array([]), minLineLength=1, maxLineGap=5)
    
    
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)
    final_output = cv2.addWeighted(img, 0.8, line_img, 1, 1)

    
    cv2.imshow("1. Canny Edges", canny)
    cv2.imshow("2. Masked Road Area", masked_edges)
    cv2.imshow("3. Final Lane Detection", final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
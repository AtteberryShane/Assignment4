import cv2
import os
import time
from ultralytics import YOLO

import numpy as np
# Re-introduce the deprecated alias so TRT’s __init__ can find it
np.bool = bool

def v11_demo():
    # Path to the TensorRT engine
    ENGINE = "yolo11n.engine"
    PT     = "yolo11n.pt"

    # Export if needed
    if not os.path.isfile(ENGINE):
       print(f"[INFO] {ENGINE} not found, exporting from {PT} …")
       YOLO(PT).export(format="engine")  

    # Now load the TensorRT engine
    model = YOLO(ENGINE, task='detect')

    # Open camera (Jetson uses GStreamer pipeline and is not same as normal webcam pipeline)
    gst = (
    "nvarguscamerasrc sensor-mode=4 ! "                               
    "nvvidconv flip-method=0 ! "                                      
    "video/x-raw(memory:NVMM),width=(int)640,height=(int)480,framerate=(fraction)30/1 ! " 
    "nvvidconv ! "                                                    
    "video/x-raw,format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=(string)BGR ! "
    "appsink drop=1"
    )

    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: unable to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame)  
        end = time.time()
        annotated = results[0].plot()

        fps = 1 / (end - start)
        cv2.putText(annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("YOLO11", annotated)
        if cv2.waitKey(1) == 27:  # ESC 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    v11_demo()

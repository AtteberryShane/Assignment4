import cv2, os, time
from ultralytics import YOLO
import numpy as np 

# Re-introduce the deprecated alias so TRT’s __init__ can find it
np.bool = bool

def seg_demo():
    # Path to the TensorRT engine
    ENGINE = "yolo11n-seg.engine"
    PT     = "yolo11n-seg.pt"

    # Export if needed
    if not os.path.isfile(ENGINE):
        print(f"[INFO] exporting seg engine from {PT} …")
        YOLO(PT).export(format="engine")

    model = YOLO(ENGINE)  
    
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
        if not ret: break

        t0 = time.time()
        results = model(frame)  
        fps = 1 / (time.time() - t0)

        vis = results[0].plot()  
        cv2.putText(vis, f"Seg FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("YOLO11 Segmentation", vis)
        if cv2.waitKey(1) == 27: # ESC 
            break  

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    seg_demo()


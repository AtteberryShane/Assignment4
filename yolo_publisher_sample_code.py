# Sample Code for Robotics_Assignment_4
# Copyright: 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

# YOLO Publisher code for Jetson NX.

# This code will give you an introduction to coding in ROS2. This code will run as-is, but you are required to modify it for the assignment requirement and submit the source code on Canvas.

# Refer to the Lab PowerPoint materials and Appendix of Assignment 3 to learn more about coding on ROS2 and the hardware architecture of Turtlebot3.
# You have to run this code on Turtlebot3 Nvidia Jetson.
# You would need a basic understanding of Python Data Structure and Object Oriented Programming, along with ROS2 concepts, to understand this code.

# This will be a harder coding assignment compared to Milestone Assignment 3.
# We recommend doing a revision on the Milestone Assignment 3 code to get a hang of coding in ROS2.

import rclpy
from rclpy.node import Node
from std_msgs.msg import String  
import cv2
import json  
from ultralytics import YOLO

class YoloJsonPublisher(Node):
    def __init__(self):
        ###############################################################
        # TODO 1: Initialize the Node with the name 'yolo_json_publisher'
        # Hint: super().__init__('your_node_name_here')
      
        ###############################################################


      
        ###############################################################
        # TODO 2: Create a publisher that sends standard String messages
        # Hint: self.create_publisher(MessageType, 'topic_name', queue_size)
        # We want to publish a String, to '/yolo/detections_json', with a queue of 10
        
        # self.publisher_ = 
        ###############################################################
        # Load YOLOv11 base model (Not Pose, Not Seg) onto the Jetson's GPU
        self.get_logger().info("Loading YOLOv11 model on CUDA...")
        self.model = YOLO('yolo11n.pt') 
        self.model.to('cuda:0')
        
        # GStreamer pipeline for Raspberry Pi V2 Camera on Jetson CSI port
        gstreamer_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )
        
        self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            return
        ###############################################################
        # TODO 3: Create a timer that triggers your callback function to capture frames
        # Hint: self.create_timer(timer_period_in_seconds, callback_function)
        # Set it to run every 0.05 seconds (20 Hz), and call `self.timer_callback`
        # self.timer = 
  
        ###############################################################

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]

        # Initialize the dictionary to hold our data
        detection_data = {
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
            "frame_id": "camera_link",
            "detections": []
        }

        # Populate the dictionary with YOLO results
        for box in results.boxes:
            x_center, y_center, width, height = box.xywh[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            detection_data["detections"].append({
                "class_name": self.model.names[class_id],
                "confidence": confidence,
                "bbox": {"cx": x_center, "cy": y_center, "w": width, "h": height}
            })
        ###############################################################
        # TODO 4: Convert the `detection_data` Python dictionary into a JSON formatted string
        # Hint: Use the json.dumps() function
        # json_str = 
      
        ###############################################################
      
        ###############################################################
        # TODO 5: Create your ROS 2 String message and publish it
        # Hint:
        # msg = String()
        # msg.data = your_json_string
        # self.publisher_.publish(msg_variable)
      
        msg = String()
        msg.data = json_str
        ###############################################################

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
        ###############################################################
        # TODO 6: Initialize the ROS 2 Python client library
        # Hint: rclpy.init with necessary arguments
        
        ###############################################################
        node = YoloJsonPublisher()
        
        try:
            ###############################################################
            # TODO 7: Spin the node so it stays alive and continues to trigger the timer
            # Hint: rclpy.spin with necessary arguments
          
            ###############################################################
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

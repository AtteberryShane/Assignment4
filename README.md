# 2026 CS 4379K / CS 5342 Introduction to Autonomous Robotics, Robotics and Autonomous Systems

## Programming Assignment: Milestone 4 (V1.0)

**Minhyuk Park and Tsz-Chiu Au**

### Introduction

Welcome to CS 7389K. The following milestones will give you an idea of how to interact with ROBOTIS’ Turtlebot3 Waffle Pi with a Manipulator Arm using the Robot Operating System 2.

In this fourth milestone, you will learn how to use YOLO, a real-time object detection algorithm based on a fully Convolutional Neural Network. YOLO’s input is a camera image fed in real time from a Raspberry Pi v2 camera, and the output would be a bounding box showing which part of the image belongs to which object, with a text label. You will find the tracking accuracy of YOLO to be acceptable, but you will find that labeling may not always be correct. This is because the generic YOLO model is trained on the Microsoft COCO dataset.

To do this, you will deploy our pre-configured Docker container that sets up all the software that is required for the assignment.
Please refer to the following video for an explanation of what a Docker container environment is. 

[https://www.youtube.com/watch?v=Gjnup-PuquQ](https://www.youtube.com/watch?v=Gjnup-PuquQ)

Robot Operating System version is associated with Ubuntu Long-Term Support Versions (e.g. Ubuntu 22.04 with Humble). We are using **ROS 2 Humble in a Docker environment** for Remote-PC. You might find the official tutorial on ROS 2 Humble useful in this course:

[https://docs.ros.org/en/humble/Tutorials.html](https://docs.ros.org/en/humble/Tutorials.html)

For all questions regarding milestone assignments and the robot, **you should contact the Doctoral Instructor Assistant via direct message on Slack**. Please do not contact the Instructor with questions regarding the milestone assignments. This is the URL for Slack for this course. 

<https://spring2026txstrobot.slack.com/>

Please head here for an introduction to the YOLO Computer Vision Object Detection Algorithm.
[https://docs.ultralytics.com/](https://docs.ultralytics.com/)

### Assignment requirement 

Assignment 4 requires a physical demonstration in front of Professor Tsz-Chiu Au. Please arrange a time with him for a physical demonstration. You need to demonstrate that you can integrate what you have learned about ROS2 and turtlebot and YOLO for missions outlined in milestone 4. The grading for this assignment will be divided into three parts. Please submit the source code of your program after the demo.

You need to demonstrate that you can utilize YOLO for your needs based on the requirements. 

Refer to the demo requirement section at the end of the milestone assignment on what to include in the demo. Rules for robot usage will apply for working with the physical Turtlebot3. Please refer to the inventory list given to you separately.

> **Major Changes**
> * v 1.0  Initial public release

---

### Objects Provided

In order to complete the demonstration requirements outlined below, we are providing you with a few objects. The objects provided are shown below.

* Bottle
* Bear Doll
* Computer Mouse

These object was chosen after testing with generic MS COCO trained yolo v11 on various angles and distances for the most stable inference results. You will find that under certain conditions, YOLO can detect but mislabel objects. You might have to consider multiple labels from YOLO’s COCO dataset to be a particular object of interest (e.g., a Teddy bear can be detected as a donut if the camera view is not just right. You can have some logic, such as if teddy bear and donut, this is a bear doll).

### Part 1: Object Detection on Jetson with CUDA 

Part 1 will show you how to run YOLO, a popular object detection and localization algorithm on the Jetson NX. Jetson allows developers to start developing Artificial Intelligence applications on Edge devices with familiar tools present on a normal desktop environment for Nvidia GPUs.

To quote Nvidia, Jetson Xavier NX delivers up to 21 TOPS, making it ideal for high-performance compute and AI in embedded and edge systems. You get the performance of 384 NVIDIA CUDA® Cores, 48 Tensor Cores, 6 Carmel ARM CPUs, and two NVIDIA Deep Learning Accelerators (NVDLA) engines. Combined with over 59.7GB/s of memory bandwidth, video encode, and decode, these features make Jetson Xavier NX the platform of choice to run multiple modern neural networks in parallel and process high-resolution data from multiple sensors simultaneously.

This assumes that you have a working setup from Milestone Assignment 1 Part 1. Please execute all instructions with **[Remote PC]** on Docker shell. Note that you have to enable GUI and start the Docker container by following instruction from Milestone Assignment 1. Please execute all instructions with **[Turtlebot Nvidia Jetson]** on Turtlebot Jetson's native bash shell without Docker.

The following instructions are from our own and not from online sources. Please follow carefully and ask the IA for assistance should you face any problems.

1. **[Turtlebot Nvidia Jeston]** On Jetson, you can look at CPU and GPU usage in real-time by using the following command on a separate terminal window. CPU will show real-time CPU load, and GR3D_FREQ will give you an idea for real-time GPU load. You can run this on another terminal window.
```bash
tegrastats

```


2. **[Turtlebot Nvidia Jeston]** While connected to the internet, git clone and download the demo in Turtlebot's Nvidia Jetson. 

```bash
cd ~/
git clone https://github.com/dmz44/Robotics_Assignment_4.git

```


3. **[Turtlebot Nvidia Jeston]** Go to the folder in a new terminal window
```bash
cd ~/Robotics_Assignment_4/Assignment_4_demo/CUDA_Demo
```


4. **[Turtlebot Nvidia Jeston]** Run yolo v11 demo and point the camera in front of the robot at various objects. You should see real-time bounding boxes and class labels.
```bash
python3 yolov11_demo.py

```


5. **[Turtlebot Nvidia Jeston]** Run yolo v11 seg demo and point the camera in front of the robot at various objects. You should see real-time bounding boxes, class labels, and binary masks of each object showing which object each pixel belongs to.
```bash
python3 yolov11_seg_demo.py

```


6. **[Turtlebot Nvidia Jeston]** Run yolo v11 pose demo and point the camera in front of the robot at people. You should see real-time human skeleton detection.
```bash
python3 yolov11_Pose_demo.py

```



**[Optional][Turtlebot Nvidia Jeston]** YOLO has different sized models with different computational requirements for real-time performance. We encourage you to experiment with different models for each script by replacing the names for engine and pt with different model names corresponding to size of the models. Listed from smallest to largest, these models trade inference speed and accuracy. After you replace the names, the Ultralytics package will automatically download and generate relevant files for you. Note that your frame per second would be bottlenecked by the camera’s sensor mode.

Relevant code section. Please change names of both pt and engine files.

```python
ENGINE = "yolo11n.engine"
PT     = "yolo11n.pt"

```

**YOLO v11 base**

```text
yolo11n.pt    yolo11n.engine
yolo11s.pt    yolo11s.engine
yolo11m.pt    yolo11m.engine
yolo11l.pt    yolo11l.engine
yolo11x.pt    yolo11x.engine

```

**YOLO v11 seg**

```text
yolo11n-seg.pt    yolo11n-seg.engine
yolo11s-seg.pt    yolo11s-seg.engine
yolo11m-seg.pt    yolo11m-seg.engine
yolo11l-seg.pt    yolo11l-seg.engine
yolo11x-seg.pt    yolo11x-seg.engine

```

**YOLO v11 pose**

```text
yolo11n-pose.pt    yolo11n-pose.engine	
yolo11s-pose.pt    yolo11s-pose.engine	
yolo11m-pose.pt    yolo11m-pose.engine	
yolo11l-pose.pt    yolo11l-pose.engine	
yolo11x-pose.pt    yolo11x-pose.engine	

```

---

### Part 2 Object Detection on Jetson with NPU

Part 2 will show you how to run YOLO on Hailo 8 NPU. Hailo 8 can be installed on any edge computer with support for a PCI-E 4x interface. This means it can be installed at a slot normally reserved for NVME SSDs and achieve real-time performance.

To quote the manufacturer, the Hailo-8 edge AI processor, featuring up to 26 tera-operations per second (TOPS), significantly outperforms all other edge processors. Its area and power efficiency are far superior to other leading solutions by a considerable order of magnitude, at a size smaller than a penny, even including the required memory. With an architecture that takes advantage of the core properties of neural networks, Hailo 8 neural chip allows edge devices to run deep learning applications at full scale more efficiently, effectively, and sustainably than other AI chips and solutions, while significantly lowering costs.

The following instruction was adopted from Hailo 8 examples git repository. The provided code is modified code from the following git that takes in Pi v2 camera feed.
[https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/hailo-8/python/object_detection](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/hailo-8/python/object_detection)

This assumes that you have a working setup from Milestone Assignment 1 Part 1. Please execute all instructions with **[Remote PC]** on Docker shell. Note that you have to enable GUI and start the Docker container by following instruction from Milestone Assignment 1. Please execute all instructions with **[Turtlebot Nvidia Jetson]** on Turtlebot Jetson's native bash shell without Docker.

1. **[Turtlebot Nvidia Jeston]** Whenever you are running an application using the Hailo accelerator, you can check the status of the Hailo accelerator using this command on another terminal window.
```bash
hailortcli monitor

```


2. **[Turtlebot Nvidia Jeston]** Assuming you have downloaded the files, go to the hailo_demo folder in a new terminal window.
```bash
cd ~/Robotics_Assignment_4/Assignment_4_demo/Hailo-Application-Code-Examples/runtime/hailo-8/python/object_detection

```


3. **[Turtlebot Nvidia Jeston]** Open a new terminal and set the following environment variable to enable hailortcli monitor to monitor on that particular terminal.
```bash
# 1. Set the environment variable for this terminal session
export HAILO_MONITOR=1

```


4. **[Turtlebot Nvidia Jeston]** Run yolo v8 demo on the terminal window you just export  and point the camera in front of the robot at various objects. You should see real-time bounding boxes and class labels.
```bash
python3 object_detection.py -n yolov8s.hef -i camera

```



**[Optional][Turtlebot Nvidia Jeston]** We encourage you to experiment with different models for each script by replacing the names for engine and pt with different model names corresponding to size of the models.

**YOLO v11 base**

yolo11n

```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov11n.hef
python3 object_detection.py -n yolov11n.hef -i camera

```

yolo11s

```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov11s.hef
python3 object_detection.py -n yolov11s.hef -i camera

```

yolo11m

```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov11m.hef
python3 object_detection.py -n yolov11m.hef -i camera

```

yolo11l

```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov11l.hef
python3 object_detection.py -n yolov11l.hef -i camera

```

yolo11x

```bash
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8/yolov11x.hef
python3 object_detection.py -n yolov11x.hef -i camera

```

---

### Part 3: Programming a publisher and a subscriber for YOLO

In order to complete part 3, you need to learn how to use ROS 2 to command the turtlebot using your own program. Refer to the Assignment 3 Appendix for help regarding this process. Take a look at the provided publisher and subscriber script for YOLO in the git repository.

We are giving you sample codes, one for the subscriber and one for the publisher, to get you started.

---

### Demo Requirements (10-15 Minute Demonstration) - 100 points

We will allow you to have multiple attempts for each demo. For additional accommodations necessary, please contact IA or the professor.
Your submission must include a single .zip file containing all of your source code.

**Part A: YOLOv11 Demonstration on Jetson GPU -20 points**
This part demonstrates your ability to run various YOLOv11 models on the Jetson's GPU using a live camera feed.

* **Object Detection (YOLOv11):** Run the standard YOLOv11 detection model. Point the camera at several different objects (e.g., a person, a bottle, a laptop) and show the model drawing accurate bounding boxes and class labels in real-time.
* **Pose Estimation (YOLOv11-Pose):** Run the pose estimation model. Have a group member step in front of the camera and show the model successfully detecting the person and overlaying the pose skeleton (the lines connecting joints).
* **Segmentation (YOLOv11-Seg):** Run the segmentation model. Point the camera at multiple objects and show the model generating distinct color masks for each detected instance, demonstrating pixel-level recognition.
* **Prove CUDA Usage:** This is a critical step. While the YOLOv11 models are running, open a new terminal and run `tegrastats`. You must show the output of `tegrastats`, which should indicate that the CUDA cores are active and under load, confirming that inference is running on the CUDA cores.

**Part B: YOLO Base Demonstration on Hailo NPU -10 points**
This section proves you can offload inference to the Hailo-8 NPU for acceleration.

* **Launch Model:** Run any YOLO object detection model that is specifically compiled for the Hailo-8 NPU.
* **Demonstrate Detection:** Show the model performing real-time object detection on the live camera feed, similar to Part A.
* **Prove NPU and SOC Usage:** While the YOLO model is running, open a new terminal and run a Hailo utility command `hailocli monitor`. You must show the output of this command along with `tegrastats`, which should indicate that the Hailo NPU is active and under load while Jetson systems are under relatively low load, confirming that inference is running on the accelerator.

**Part C: ROS2 Publisher & Subscriber Demonstration -  -70 points**

This part showcases your ability to integrate the given ROS2 code for a robotic task.

Slides detailing the arena and detailed rules will be released separately.

* a): Demonstrating Visual Servoing to a Bottle (Task 1) -10 points

For part C's a), you need to code a visual servoing code that turns the wheels of the Turtlebot 3 so that the front of the robot faces the target object, a bottle.

Demonstrate your code by moving the bottle slowly in front of the robot. The robot should lock onto the bottle using the front camera.

To complete this requirement, you would need to adapt milestone 3’s teleoperation code and milestone 4’s code for YOLO and integrate them into the code for both the remote PC and Jetson. We recommend using SSH to access the Turtlebot’s Jetson, since the number of Flipbooks is limited.

* b): Pick the bottle with Turtlebot (Task 1) -10 points

For part C's b), you need to extend a)’s code so that the Turtlebot 3 drives to the bottle, faces the bottle, extends the arm and grab onto the bottle. You can use non-precise ways to achieve this, for example, by applying forward velocity on the wheels for a set time in an open-loop fashion.

To complete this requirement, you would need to adapt milestone 3’s teleoperation code and milestone 4’s code for YOLO and integrate them into the code for both the remote PC and Jetson. We recommend using SSH to access the Turtlebot’s Jetson, since the number of Flipbooks is limited.

* c): Pick and Place with Turtlebot (Task 2) -20 points

For part C's b), you need to extend a)’s code so that the Turtlebot 3 returns to the base after picking up the bottle, and release the bottle on the ground at the home base near the robot's starting location. You can use non-precise ways to achieve this, for example, by applying forward velocity on the wheels for a set time in an open-loop fashion.

To complete this requirement, you would need to adapt milestone 3’s teleoperation code and milestone 4’s code for YOLO and integrate them into the code for both the remote PC and Jetson. We recommend using SSH to access the Turtlebot’s Jetson, since the number of Flipbooks is limited.

* d): Pick and Place two bottles with Turtlebot (Task 2) -30 points

For part C's d), you need to extend c)’s code so that the Turtlebot 3 repeats the pick and place operation for a total of two bottles. You can use non-precise ways to achieve this, for example, by applying forward velocity on the wheels for a set time in an open-loop fashion.

To complete this requirement, you would need to adapt milestone 3’s teleoperation code and milestone 4’s code for YOLO and integrate them into the code for both the remote PC and Jetson. We recommend using SSH to access the Turtlebot’s Jetson, since the number of Flipbooks is limited.

---

### Appendix

#### [Optional] Running YOLO v11 on Your Desktop Machine

You can run YOLO v11 with either CPU only or CUDA on your own desktop machine with any architecture, with the following setup and modifications. This capability might help you with offloading some of the development for this assignment to your own machine. However, while this instruction was tested on our machines, we would not offer official support for you running YOLO on your own machine. We would also not provide ways to hook up camera feed from Gazebo’s simulated turtlebot onto YOLO v11.

Recommended reading: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

**[Remote PC]** It is not recommended to install YOLO in a Python virtual environment, such as Conda, if you want it to work well with ROS2 or other binary programs installed through apt-get. This is the primary reason why we did not install a Python virtual environment on Jetson.

You can deactivate your Python virtual environment temporarily and install Ultralytics packages if you need your virtual environment.

**[Remote PC]** Install GPU drivers such as CUDA and ROCm, Pytorch, and TorchVision. When installing PyTorch and TorchVision, please follow the exact instructions on the official website. [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) .

**[Remote PC]** On your terminal, install the Ultralytics package and OpenCV

```bash
pip install ultralytics opencv-python

```

**[Remote PC]** Here is an example YOLO v11 Python code adapted to a desktop environment. We have changed cap interface to webcam for you. Replace `PT_MODEL` with “yolov11-seg.pt” or “yolov11-pose.pt” to convert it to seg or pose models. Refer to above for full list of model names and associated .pt names.

**Desktop YOLO v11 demo code on CPU or CUDA**

```python
import cv2
import time
from ultralytics import YOLO
import numpy as np

def desktop_demo():

    PT_MODEL = "yolov11n.pt"  # Change this line for different models
    model = YOLO(PT_MODEL)
   
    # Capture through webcam
    cap = cv2.VideoCapture(0) # Replace with the name of a photo or video if needed
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
        annotated_frame = results[0].plot()

        # Calculate and display FPS
        fps = 1 / (end - start)
        cv2.putText(annotated_frame,
                     f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("YOLOv11 Desktop Demo", annotated_frame)
        
        if cv2.waitKey(1) == 27: #ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    desktop_demo()

```

#### [Optional Reading] Platform Specific Optimizations on Jetson for Real-time Performance

You might have noticed that the provided code is different from the normal code that you are used to running in other computer science classes. Jetson is an edge platform that requires optimization for real-time performance and requires some specialized ways to perform everyday functions.

**Camera connection for Jetson**
The first major difference is how we connected the camera to the Jetson. Instead of using a USB connection to a normal webcam, we use the MIPI CSI (Camera Serial Interface) port on the Jetson to connect to a camera at a lower level. The CSI port is a direct, high-bandwidth connection to the Jetson's dedicated Image Signal Processor (ISP).
By using the CSI port, the raw image data bypasses the slow, general-purpose USB controller and goes straight into the ISP. The ISP handles critical pre-processing tasks like de-Bayering (converting raw sensor data to a color image), color correction, and noise reduction using dedicated hardware. This frees the CPU from doing any of this initial heavy lifting and, most importantly, achieves lower photon-to-response latency.
In practice, this changes how the camera is accessed through code. The gst string in your code is actually a command that builds a hardware-accelerated video processing pipeline using GStreamer. More specifically, we use nvarguscamerasrc, which is a GStreamer plugin specifically designed for cameras connected to the MIPI CSI (Camera Serial Interface) port on the Jetson.
Let us break down the pipeline illustrated in gst string in more detail. As mentioned, nvarguscamerasrc captures video from the CSI/ISP hardware. video/x-raw(memory:NVMM) is arguably the most important performance trick in the pipeline. NVMM (NVIDIA Managed Memory) tells the system to keep the video frame entirely within the GPU/SoC's memory space. The frame is never copied to the CPU's main RAM. This concept, often called "zero-copy," is a massive performance win because moving large video frames between CPU and GPU memory is a slow, performance-killing bottleneck. nvvidconv is a hardware-accelerated video converter. Tasks like resizing the video (e.g., from 1080p to 480p) or changing the color format (e.g., from the camera's native format to the BGR format needed by OpenCV) are performed by the Jetson's dedicated video engine, not the CPU.
Therefore, by making Jetson-specific modifications to the Python code, your Python code can have all the capturing, pre-processing, and format conversion already been done efficiently in hardware. This means that by the frame is available from appsink, the frame would already be waiting in a GPU-accessible memory location for your use.

**Camera Sensor Modes for Raspberry Pi v2 camera**
The Raspberry Pi Camera Module v2 (IMX219) has several sensor modes available on the NVIDIA Jetson Xavier NX, but they are often grouped into a few commonly used configurations accessible through the driver. The specific modes you can use depend on the camera driver (like nvarguscamerasrc for GStreamer) and the software you're using. Using different sensor modes allows you to access outputs of the camera sensor with different frame rate, field of view and bit depth. The following figure illustrates this concept very well, but with a different Single Board Computer, Raspberry Pi. This means Jetson does not have the exact sensor modes that this figure illustrates.

On Jetson, the following modes are accessible for user’s applications.

* **Mode 0 (3264x2464 @ 21 FPS):** This is the full-resolution mode, capturing the entire 8-megapixel sensor area. It's best for high-quality still images but has a lower frame rate.
* **Mode 2 (1920x1080 @ 30 FPS):** This is a standard Full HD video mode. It uses the full sensor width and then crops or scales it down, providing a wide field of view. This is a very common mode for video applications.
* **Mode 4 (1280x720 @ 120 FPS):** This is the high frame rate mode, ideal for slow-motion capture. It achieves this speed by using 2x2 binning on the sensor, which groups pixels together. This results in a "partial" or cropped field of view compared to the other modes.

**OpenCV for Jetson**
While the OpenCV you are using is similar to the OpenCV you are using for other computer science classes, we needed to compile from source to exploit the hardware on the Jetson. This is because a binary install of OpenCV from pip is CPU-only and is not configured well.
In particular, you need to enable particular sets of plugins during source compilation. We leave some of them here. A good optional exercise for the reader is to find out what each flag does and how it fits within our robot assignment. Note: Jetson Xavier series has 7.2 for CUDA_ARCH_BIN.

```bash
cmake \
…
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN='7.2' \ 
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=1 \
-D WITH_GSTREAMER=ON \
-D WITH_LIBV4L=ON \
-D ENABLE_NEON=ON \
…

```

**Jetson AI Inference and TensorRT Exploitation**
Let's now look at part of the code dealing with .pt file and .engine file.
A .pt file is a standard model from the PyTorch framework. It's flexible and contains the model's architecture and weights. When you run it, the PyTorch framework interprets this file to perform the calculations. It's portable but not optimized for any specific hardware.
An .engine file is the result of taking the .pt blueprint and compiling it with NVIDIA TensorRT SDK. A wrapper from Ultralytics takes care of the specifics in utilizing TensorRT. TensorRT is an optimizer that aggressively modifies the model to run as fast as possible on a specific NVIDIA GPU—in this case, your Jetson's integrated GPU.
This "export" step performs several key optimizations using the TensorRT SDK:

* **Layer Fusion:** It combines multiple simple AI layers (e.g., a convolution, a bias, and an activation) into a single, highly efficient custom operation. By fusing multiple computational operations into a single operation, we can achieve higher inference speed. We also set up so that during the inference process, we exploit Nvidia’s Tensor Cores, a dedicated part of the silicon that supports operations such as Fused Multiply Add, a common operation during AI inference.
* **Suggested Reading:** [https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
* **Precision Calibration:** Also known as quantization, it can convert calculations from slow 32-bit numbers to much faster 16-bit (FP16) or 8-bit (INT8) numbers, often with little to no loss in accuracy.
* **Kernel Auto-Tuning:** It selects the fastest possible algorithms for your specific Jetson GPU.

In order to do this, you need some setup in addition to the Jetpack software that contains the TensorRT SDK. That part was done by the IA.

**Pytorch and TorchVision for Jetson**
The pre-compiled versions of PyTorch available on the standard Python Package Index (PyPI) are built for desktop computers, which almost always use an x86_64 CPU architecture (from Intel or AMD). The NVIDIA Jetson, however, uses an ARM64 CPU architecture, similar to what's in modern smartphones or Apple Silicon Macs. Therefore, NVIDIA provides PyTorch wheels (.whl files) that have been specifically compiled for the Jetson's ARM64 architecture so they can run correctly.
You also need to ensure compatibility with Jetpack. The Jetson's operating system, a customized Ubuntu operating system called JetPack, includes specific versions of CUDA, cuDNN, and TensorRT that are tailored for its mobile, power-efficient GPU. NVIDIA’s custom PyTorch builds are specifically compiled and linked against these exact libraries. This ensures a stable and high-performance bridge between the PyTorch framework and the GPU hardware. Also, during compilation, NVIDIA enables specific flags and optimizations that take advantage of the unique features of the Jetson's GPU, which are different from a desktop GPU like an RTX 4090.
Therefore, when installing PyTorch and TorchVision for Jeston, you need to match your Python version and your specific Jetpack version. This effectively means you would be installing an older version of Pytorch and TorchVision on Jetson, and problems with that, such as an older version of Numpy. You need to make sure which version works with which version when working with Jetson. That part was done for you by the IA.

#### [Optional Reading] Hailo 8 and NPUs for Edge AI

**Hailo 8 Architecture and Compiler**
Along with Jetson’s onboard GPU and CUDA capabilities, we are giving you access to Neural Processing Unit Hailo 8 as well. An NPU, or Neural Processing Unit, is a specialized hardware chip designed to efficiently accelerate Artificial Intelligence (AI) tasks, particularly neural network computations.
The Hailo-8 is built on Hailo’s Structure-Defined Dataflow Architecture, which is fundamentally different from the architecture of a standard CPU or GPU. This design is what allows it to achieve very high performance in a small power envelope, making it ideal for edge AI applications.
Think of a traditional processor (like a CPU or GPU) as a large, centralized workshop with powerful tools (the ALUs) and a big warehouse for materials (the RAM/cache). To perform a task, you constantly have to fetch materials from the warehouse, bring them to a tool, process them, and then send them back to the warehouse. This back-and-forth movement of data consumes a lot of time and energy.
The Hailo-8's dataflow architecture works more like a physical assembly line. Instead of a central memory warehouse, the processing resources and memory are distributed across the chip. When a neural network is compiled for the Hailo-8, it's physically mapped onto this hardware assembly line. Data (like an input image) enters at one end, flows sequentially through the different processing stations (the hardware blocks for convolutions, activations, etc.), and the final result comes out the other end.
This approach drastically minimizes data movement, which is the key to its efficiency.
The Hailo-8's chip is made up of a grid of specialized hardware blocks:

* **Compute Cores:** These are not general-purpose cores like on a CPU. Each core is a small, specialized engine designed to perform the mathematical operations common in neural networks, primarily multiply-accumulate (MAC) operations. They are the "workers" at each station on the assembly line.
* **Distributed Memory Elements:** Instead of a large, shared L2/L3 cache, memory is broken up into many small, local SRAM blocks distributed across the chip, right next to the compute cores. This means each "worker" has a small tray of materials right at their station, eliminating the need to walk back and forth to a central warehouse.
* **Control and Data Fabric:** This is a high-speed network-on-chip that acts as the "conveyor belt" of the assembly line. It is responsible for efficiently streaming data from one compute core and its local memory to the next, as defined by the structure of the neural network.
* **Host Interface:** This component, typically using PCIe, manages communication with the host system (like a Jetson). It's responsible for receiving the input data from the host and sending the final inference results back.
To exploit this dedicated hardware, you need a specialized compiler. The Hailo Dataflow Compiler is the software that takes a standard neural network (from TensorFlow, PyTorch, etc.) and figures out the most efficient way to map it onto the Hailo-8's physical hardware. It analyzes the network graph and decides:
* Which compute cores will handle which layers of the network?
* How to partition the model's weights and activations into the distributed memory blocks.
* How to configure the data fabric (the "conveyor belt") to ensure data flows smoothly from one processing stage to the next.
The output of this compiler is the .hef file, which is not just the model's weights but a complete blueprint for configuring the entire chip to execute that specific network with maximum efficiency.

**Hailo 8 AI Inference on Python**
The Python code is structured into three main stages that run in parallel using separate threads to exploit the .hef file from the compiler.

* **Pre-processing:** This stage reads video frames from a camera or file, prepares them for the AI model (resizing, color conversion), and puts them into an input_queue.
* **Inference:** This stage, handled by the Hailo-8, takes the prepared frames from the input_queue, performs object detection, and places the results into an output_queue.
* **Post-processing:** This stage takes the inference results and the original video frames from the output_queue, draws bounding boxes on the frames, and displays them on the screen.
In particular, asynchronous inference engine,

```python
hailo_inference = HailoAsyncInference(
    net_path, input_queue, output_queue, batch_size, send_original_frame=True
)

```

creates an instance of the Hailo inference engine. It's the main bridge between your Python application and the Hailo hardware. It automatically handles loading the optimized .hef file onto the NPU,managing the data queues (input_queue, output_queue), continuously feeding image data from the input queue to the Hailo-8 over the PCIe bus, and keeping the NPU's processing pipeline full to achieve maximum throughput. This is initiated by the following line.

```python
hailo_inference.run()

```

This allows neural network calculations to be offloaded to Hailo 8 NPU, while Jetson’s CPU and GPU handle the data pipeline and user interface.

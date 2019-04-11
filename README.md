# Stereo Object Detection
Combines object detection and stereo vision to tell where and how far something is

## Requirements
* **Python 3**  
* **Ubuntu 16.04** although it should work for other versions  
* **CMake >= 3.8**  
* **(Recommended) CUDA 10.0** for GPU  



## Installation

**Install Required dependencies**  
```
$ sudo apt-get update
$ sudo apt-get install build-essential python3 python3-dev python3-pip -y
```
**Build darknet (for object detection)**  
If you don't have a GPU, omit the `GPU=1` flag
```
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make LIBSO=1 GPU=1
```

**Install Python dependencies**  
```
pip3 install -r requirements.txt
```

## Usage
**Stereo + YOLO (coming soon)**
```
python3 cam_demo.py
```

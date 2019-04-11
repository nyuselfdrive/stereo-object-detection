# Stereo Object Detection
Combines object detection and stereo vision to tell where and how far something is


## Installation

**Install Required dependencies**  
```
$ sudo apt-get update
$ sudo apt-get install build-essential python3 python3-dev -y
```
**Build darknet (for object detection)**  
If you don't have a GPU, omit the `GPU=1` flag
```
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make LIBSO=1 GPU=1
```


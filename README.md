# YOLOv5-LibTorch
Real time object detection with deployment of YOLOv5 through LibTorch C++ API

### Environment

- Ubuntu 18.04
- OpenCV 3.2.0
- LibTorch 1.6.0
- CMake 3.10.2

### Getting Started

1. Install OpenCV.

   ```shell
   sudo apt-get install libopencv-dev
   ```

2. Install LibTorch.

   ```shell
   wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
   unzip libtorch-shared-with-deps-latest.zip
   ```

3. Edit "CMakeLists.txt" to configure OpenCV and LibTorch correctly.

4. Compile and run.

   ```shell
   cd build
   cmake ..
   make
   ./../bin/YOLOv5LibTorch
   ```

Note: COCO-pretrained YOLOv5s model has been provided. For more pretrained models, see [yolov5](https://github.com/ultralytics/yolov5).
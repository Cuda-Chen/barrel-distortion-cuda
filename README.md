# barrel-distortion-cuda
A barrel distortion example written with OpenCV C++ API and CUDA.

# Dependencies
* CMake
* OpenCV
* CUDA

# How to Compile and Run
If you are using UNIX-like system type these commands:
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
$ ./barrel_distort_cpp <input image path> <K: coefficient of barrel distortion> [x corrdinate of center of distortion (in pixel)] [y corrdinate of center of distortion (in pixel)]
```

# Reference
The work of barrel distortion is originated from 逍遙文工作室. <br>
You can find his work in the following link (Chinese website):
* https://cg2010studio.com/2012/01/03/opencv-%E6%A8%A1%E6%93%AC%E9%AD%9A%E7%9C%BC%E9%8F%A1%E9%A0%AD-simulate-fisheye-lens/

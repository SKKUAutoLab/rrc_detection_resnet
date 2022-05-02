 #!/bin/bash 

/usr/bin/g++-6 *.h *.cpp -o FreeRoadDetection -I/usr/include/opencv4/ -L/usr/lib/x86_64-linux-gnu/  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_features2d -lopencv_calib3d -lpthread

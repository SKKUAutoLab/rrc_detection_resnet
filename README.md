### Introduction
This Github repository contains the source codes for training and running inference with the ResNet-based Recurrent Rolling Convolution (ResNet-RRC). The network is based on the Caffe Framework. It is related to the paper "A Deep Learning Framework for Robust and Real-time Taillight Detection under Various Road Conditions".

### Installation
0. Setup the environment for Python programming. Make sure that you are using Python 2.7. Other versions of Python may NOT work. Note that this ResNet-RRC repo has been tested only under "pure" Python environment (i.e. it has not been tested under integrated development environments such as Anaconda)
1. Get the code. We will call the directory that you cloned Caffe into as `$CAFFE_ROOT`
   ```Shell
   https://github.com/SKKU-AutoLab-VSW/rrc_detection_ResNet.git
   cd rrc_detection_ResNet
   ```
2. Before building the Caffe code in this repo, it should be noted that this repo uses the customized Caffe Framework based on the old version of the framework. You may follow [Caffe instructions](http://caffe.berkeleyvision.org/installation.html) to get general ideas on how to install, but be sure to follow the instructions mentioned in this repository.

   Install the following packages first using apt-get:

   ```Shell
   sudo apt-get install libopencv-dev
   sudo apt-get install libhdf5-serial-dev libleveldb-dev liblmdb-dev libsnappy-dev
   sudo apt-get install libgoogle-glog-dev
   ```
   
   Install protobuf 3.5.1 using pip:
   ```Shell
   pip install protobuf==3.5.1
   ```
   Note that you may also need to install the protobuf library in the system-wide environment (i.e. the sources for protobuf 3.5.1 should be manually downloaded, compiled, and installed.) Make sure that the version installed via pip and the version installed in the system-wide environment matches with each other.
   
   For libboost, version 1.65 is required (here we recommend 1.65.1), but apt-get installation will NOT resolve this, since the apt-get package lacks support for working on c++11 syntax, wheras ResNet-RRC needs c++11 syntax. So we need custom build of libboost 1.65 w/ std=c++11 option enabled and install it.

   ```Shell
   ./bootstrap.sh
   sudo ./b2 install --toolset=gcc cxxflags="-std=c++11"
   ```

   if libboost_system is not found in typical directories like `/usr/lib`, you need to add additional library path:
   ```Shell
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
   ```
   
   Install OpenCV for Python. To accurately replicate the detection speed, we recommend installing OpenCV 3.1.0. Nevertheless, you may install versions up to 3.4.8.29 if you don't mind about detection speed.
   
   Install NVIDIA [CUDA](https://developer.nvidia.com/cuda-downloads) and cuDNN(https://developer.nvidia.com/cudnn) from the official website. Note that ResNet-RRC is tested on CUDA versions up to 10.1 and cuDNN versions up to 7.6.5. **Beware that higher versions of CUDA + cuDNN will NOT work.**
   
   Write down `Makefile.config` according to your Caffe installation. An example file is given as `Makefile.config.example`.
   Caffe framework for ResNet-RRC will build only with GCC versions 7.0 or lower. If your GCC version is higher than that, you need to downgrade it.
   Perform the code build using `make` commands:
   ```Shell
   make -j32
   export PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python
   make py
   make test -j8
   make runtest -j8
   ```

   If you encounter any errors in building, please refer to TROUBLESHOOTING.md for details.

### Preparation
1. Check the existance of pre-trained ResNet-18.
   By default, we assume the model is stored in `$CAFFE_ROOT/models/ResNet/`.
   You can download it from [this link](https://onedrive.live.com/?authkey=%21AJiz7yGu%5F5i4iBw&cid=7C725726AF404CFD&id=7C725726AF404CFD%21451&parId=root&o=OneUp)

2. To train and test ResNet-RRC on KITTI, we need to prepare the KITTI dataset first.
   Download the KITTI dataset(http://www.cvlibs.net/datasets/kitti/eval_object.php).
   By default, we assume the data is stored in `$HOME/data/KITTI/`
   Unzip the training images, testing images and the labels in `$HOME/data/KITTI/`.

3. Create the LMDB file.
   For training, the KITTI labels should be converted to VOC type. Run the label converter:
   ```Shell
   python ./convert2xml_RRC.py
   ```
   VOC type labels will be generated in `$HOME/data/KITTI/training/labels/xml/`.
   
   Next, create trainval.txt, test.txt, and test_name_size.txt in $CAFFE_ROOT/data/KITTI/:
   ```Shell
   cd $CAFFE_ROOT/data/KITTI/
   ./create_list.sh
   cd ../..
   ```
   
   Generate LMDB Database files.
   ```Shell
   ./create_data.sh
   ```
   You can modify the parameters in create_data.sh if needed.
   It will create lmdb files for trainval and test with encoded original image:
      - $HOME/data/KITTI/lmdb/KITTI_training_lmdb/
      - $HOME/data/KITTI/lmdb/KITTI_testing_lmdb/
   and make soft links at data/KITTI/lmdb
   
   We need to build the Lane Module of the Taillight Pipeline. execute the following commands to build the sources:
   ```Shell
   cd LaneDetection
   /usr/bin/g++-6 *.h *.cpp -o FreeRoadDetection -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_features2d -lopencv_calib3d -lpthread
   chmod 777 FreeRoadDetection
   ```
   If you are using OpenCV 4.0 or higher, use the following commands to build the sources:
   ```Shell
   cd LaneDetection
   /usr/bin/g++-6 *.h *.cpp -o FreeRoadDetection -I/usr/include/opencv4/ -L/usr/lib/x86_64-linux-gnu/  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_video -lopencv_features2d -lopencv_calib3d -lpthread
   chmod 777 FreeRoadDetection
   ```
   
### Training
   To train your model for car detection on some dataset (e.g. KITTI), execute the following:
   ```Shell
   python examples/all/rrc_train.py
   ```
   You can modify the parameters such as dataset names if needed. Note that the script itself becomes trained on not only cars, but also other objects in traffic such as pedestrians and cyclists.
   
   For training on taillights, after preparing ground truth labels for taillights in images, execute the following:
   ```Shell
   python examples/TL/rrc_train.py
   ```
   Here the input images and labels can come from:
   - Original KITTI dataset
   - KITTI dataset with images cropped by ego-lane regions
   - KITTI dataset with images cropped by individual car regions

   We train our models in a computer with 4 TITAN Xp(Maxwell) GPU cards. If you only have one GPU card, you should modify the script `rrc_train.py`.
   ```Shell
   line 118: gpus = "0,1,2,3" -> gpus = "0"
   ```
   If you have two GPU cards, you should modify the script `rrc_train.py` as follows.
   ```Shell
   line 118: gpus = "0,1,2,3" -> gpus = "0,1"
   ```
### Inference
   Before performing inference, you need to have your own trained weights. If you want to skip the training process and directly use pretrained models, download the weights and prototxt definitions from the following links:
   
   - Pretrained weights for Car Module: [caffemodel](https://drive.google.com/file/d/1C-k1pFW37bA-sarLexewF-vdFlImonf8/view?usp=sharing), [solverstate](https://drive.google.com/file/d/1ub6q3WGBLQjtiL4tszwZ0pveYYmTK7Fy/view?usp=sharing), [solver prototxt](https://drive.google.com/file/d/1E39IBlHRaZsIW0I24SgpGILbXnbOxYJM/view?usp=sharing), [test prototxt](https://drive.google.com/file/d/1xMvo_AJrOEe9gThXIa62PFfp-DFfHB2q/view?usp=sharing), [train prototxt](https://drive.google.com/file/d/1kg9ojxLd08iYQaISmsiMI0uejeUGVor1/view?usp=sharing)
   - Pretrained weights for Taillight Module: [caffemodel](https://drive.google.com/file/d/1mF0udE1-ltt8Cx79dHR5hHiGzUY5mrI6/view?usp=sharing), [solverstate](https://drive.google.com/file/d/1sN_GWVfnlL4rDmquk4wkCyaQBw-cZZdR/view?usp=sharing), [solver prototxt](https://drive.google.com/file/d/1dKmfFept66YIPsvPpmFNZvEvsCHgrGOH/view?usp=sharing), [test prototxt](https://drive.google.com/file/d/1nhGVSPLmLPWjx5rhie5-swPRujL8t9Hi/view?usp=sharing), [train prototxt](https://drive.google.com/file/d/1Mcpa9e1xUO7nqiz0pOCQICjcRZer-fqg/view?usp=sharing)
   
   When using the pretrained weights of Car Module, place the files under `$CAFFE_ROOT/models/ResNet/KITTI/RRC_2560x768`.
   When using the pretrained weights of Taillight Module, place the files under `$CAFFE_ROOT/models/ResNet/KITTI_carTL/RRC_768x768`.
   
   If you want to test the lane, car, and taillight modules separately, execute either one of the following:
   ```Shell
   cd LaneDetection
   ./FreeRoadDetection # for lane detection
   cd ..
   python examples/car/rrc_test.py # for car detection
   python examples/TL/rrc_test.py # for TL detection
   ```
   You can modify the parameters such as dataset names if needed.
   The Lane Detection Module contains various parameters for ego-lane region estimation in `LaneRegionSetting.txt`. Feel free to adjust those parameters as needed.
   
   For the Car and Taillight Detection Modules, you should modify [line 10: img_dir] to [your path to kitti testing images] if necessary.
   For testing a model you trained, you should modify the path in `rrc_test.py` as well.
   To accurately replicate the results mentioned in our paper, `examples/car/rrc_test.py` must fetch the model weights trained on the scene image cropped by ego-lane regions (or the whole scene image), and `examples/TL/rrc_test.py` must fetch the model weights trained on rear-facing cars.
   
   If you want to test the lane, car, and taillight modules simultaneously as components of the whole taillight pipeline, launch three instances of terminal process and execute each of the following:
   ```Shell
   ###################### Shell 1 #############################
   cd LaneDetection
   ./FreeRoadDetection # for lane detection
   ###################### Shell 2 #############################
   python examples/car/rrc_test_realTime.py # for car detection
   ###################### Shell 3 #############################
   python examples/TL/rrc_test_realTime.py # for TL detection
   ```
   Note that the order matters. You must execute in the order of Lane -> Car -> Taillight Module, and wait for at least 2 seconds between executing the modules. Otherwise, the pipeline may break.
   The lane detection executable and the two Python scripts assume that outputs of lane detection are stored in `$HOME/tl_temp/lane/`, and outputs of car detection are stored in `$HOME/tl_temp/car/`.
   You can modify the parameters such as dataset names if needed.
   To accurately replicate the results mentioned in our paper, `examples/car/rrc_test_realTime.py` must fetch the model weights trained on the scene image cropped by ego-lane regions (or the whole scene image), and `examples/TL/rrc_test_realTime.py` must fetch the model weights trained on rear-facing cars.
   
   

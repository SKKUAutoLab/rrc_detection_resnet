### Troubleshooting Tips

Q1. I have errors complaining about google-protobuf. What should I do?

A: For protobuf, versions up to 3.5.1 has been verified, but not the others. You should install this using the command `pip install protobuf==3.5.1`.
  If that does not work either, the source should be manually downloaded, compiled, and installed in your Python environment.
  If that still does not work, install the following packages:
```Shell
  sudo apt-get install libprotobuf-dev protobuf-compiler
```

Q2. I have errors complaining about glog. What should I do?

A: glog should be installed using `sudo apt-get install libgoogle-glog-dev`

Q3. I have errors complaining about libboost. Error persists even if I installed version 1.65. What should I do?

A: For libboost, only the version 1.65 has been verified to work for ResNet-RRC, and apt-get installation will NOT resolve this.
The apt-get package build lacks support for working on behalf of c++11 syntax. We need custom build of libboost 1.65 with std=c++11 option enabled and install it.
```Shell
./bootstrap.sh
sudo ./b2 install --toolset=gcc cxxflags="-std=c++11"
```
and if that STILL doesn't work due to complaining about libboost_system not found, execute the following.
```Shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

Q4. I have errors complaining about cuBlas. What should I do?

A: First, check libcublas10 version.
```Shell
dpkg -l | grep libcublas
```
if version is 10.2.2.89, remove it,
```Shell
sudo apt remove libcublas10
```
and then install downgraded version.
```Shell
sudo apt install libcublas10=10.2.1.243-1
```

Q5. For some reason, Caffe build fails on `detection_output_layer.cu` or `video_data_layer.cpp`. What should I do?

A: This is because you are using GCC version greater than 7.0. The GCC should be downgraded.
It is recommended to do so via apt-get, but if your system cannot reach gcc-6, do the following:

1. open `/etc/apt/sources.list`
2. add `deb http://{}.archive.ubuntu.com/ubuntu/ bionic main universe` to file. Here the `{}` denotes domain country code (e.g. 'us' for United States, 'kr' for Korea, and so on.) 
3. execute the following:
   ```Shell
   sudo apt update
   sudo apt-get install gcc-6 g++-6 -y
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
   ```
Q6. When I installed the Caffe framework from this repo and execute `rrc_train.py`, the following error occurs:

`ImportError: dynamic module does not define module export function (PyInit__caffe)`

What should I do?

A: Simply close the terminal and re-launch it. Then re-execute `rrc_train.py`


# The MRFMap library

See also: \
[http://mrfmap.github.io](http://mrfmap.github.io), the project page for MRFMap. \
[mrfmap_ros](https://github.com/mrfmap/mrfmap_ros), the ROS wrapper for this library.


Originally developed by Kumar Shaurya Shankar while working in the [RISLab](https://www.rislab.org/) at The Robotics Institute at Carnegie Mellon University with Prof. Nathan Michael.

This is research software - although efforts are made for it to be reliable, it is untested and unsuitable for critical operations.

MRFMap is released under the permissive BSD-3 license. 

Please feel free to issue pull requests and bugs.

If you end up using this library, please cite
```bibtex
@inproceedings{shankar20mrfmap,
  author = {Shankar, Kumar Shaurya and Michael, Nathan},
  title = {{MRFMap: Online Probabilistic 3D Mapping using Forward Ray Sensor Models}},
  booktitle = {Robotics: Science and Systems},
  year = 2020
}
```

# Installing

## Prerequisites
* An NVidia GPU with compute capability >5.3 (which means >Pascal series(GTX 1060+)) since we use half2 intrinsics
* CUDA version >10
* A linux machine (Although Windows builds are not too hard to accomplish - Please contact the maintainer/send a pull request if you are interested in the same!)
* CMake >3.13


Tested known-good build configurations:
* Ubuntu 16.04 + CUDA 10.1 on GeForce GTX 1070
* Ubuntu 18.04 + CUDA 10.1 on GeForce GTX 2060 Super
* Ubuntu 18.04 + CUDA 10.2 on NVIDIA Xavier NX

Other configurations (such as newer versions of CUDA/different Linux flavours) should work as well, however they haven't been tested.

## Dependencies
The MRFMap library depends on a number of third party libraries
* [Eigen 3.3.7](https://gitlab.com/libeigen/eigen/-/releases/3.3.7) (for general purpose matrix operations)
* [Octomap](https://github.com/OctoMap/octomap) (for comparison)
* [libcuckoo](https://github.com/icoderaven/libcuckoo) (for timing)
* [pybind11](https://github.com/icoderaven/pybind11) (for Python bindings)
* [pangolin](https://github.com/icoderaven/Pangolin) (for our customised OpenGL viewer)
* [gvdb-voxels](https://github.com/icoderaven/gvdb-voxels) (for the bulk of our GPU based implementation)
* CUDA (for GP-GPU computation)

Please see the respective libraries for more details on configuration options and additional dependencies. The project pulls in forks of all these libraries that are configured to correctly export build targets and compile cleanly within this project. A list of packages required on ubuntu 18.04 for all the libraries is 
```
sudo apt install libglew-dev libgl1-mesa-dev libxkbcommon-dev ffmpeg libavcodec-dev libavutil-dev \
libavformat-dev libswscale-dev libavdevice-dev libjpeg-dev libpng12-dev libtiff5-dev libopenexr-dev \
libxinerama-dev libxrandr-dev libxcursor-dev libxi-dev libx11-dev libyaml-cpp-dev
```
This library is built via modern CMake constructs, and so please ensure that you have CMake version > 3.13. You can use the official kitware repositories at [https://apt.kitware.com/](https://apt.kitware.com/) to effortlessly update your CMake on Ubuntu machines.


## Steps to build the library
In a fresh clone

```
 mkdir build && cd build
 cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
 make -j8 install 
 ```

 ### Options
 #### Python bindings
  In order to be backwards compatible with ROS python modules, we default to building python 2.7 pybind11 library modules. In case you prefer python3 modules, just add ``` -DMRFMAP_USE_PYTHON2=OFF```  to the ```cmake``` command above. In case you do not want to build python modules at all, use ``` -DMRFMAP_BUILD_PYTHON_BINDINGS=OFF```.

## Verify install

To verify that the pipeline works, at the end from the build folder, run
```
./testMRFMapViewer
```
You should be able to see the simple coffin_world example with the inference running on MRFMap and Octomap simultaneously at a fine resolution.

# Including the MRFMap library in your CMake project
This should install all the requisite install targets and libraries in the install subdirectory. To pull in the targets in your own CMake project
```
set(mrfmap_PATH "path/to/mrfmap/install/share/cmake/")
find_package(mrfmap CONFIG REQUIRED PATHS ${mrfmap_PATH})

add_executable(your_executable_name src/your_executable_source.cpp)
target_link_libraries(
  your_executable_name PRIVATE mrfmap::mrfmap mrfmap::pango_viewer)
```
You can either extract the RPATH and build your executable with it, or simply copy the lib folder .so files
```
get_target_property(SHARED_LIB_PATH mrfmap::mrfmap IMPORTED_LOCATION_RELEASE)
get_filename_component(DIR ${SHARED_LIB_PATH} DIRECTORY)
install(
  DIRECTORY ${DIR}/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  FILES_MATCHING
  PATTERN "*.so*")
``` 

An example implementation is in the [mrfmap_ros](https://github.com/mrfmap/mrfmap_ros) package, the ROS wrapper of MRFMap.

Alternatively, you can simply fetch this library within your CMake codebase, and call ```set(MRFMAP_EXPORT_NAME "${PROJECT_NAME}-targets")``` before the ```add_subdirectory()``` call to push all generated exports within your ```${PROJECTNAME_NAME}-targets``` install command.

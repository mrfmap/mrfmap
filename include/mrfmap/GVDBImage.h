#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

#include <mrfmap/GVDBParams.h>
#include <mrfmap/GVDBCommon.cuh>

#ifdef DEBUG_IMAGE
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#endif

#include <cuda.h>

#define USE_DEPTH_IMAGE_BUFFER

class GVDBImage {
 public:
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> MRow;

 private:
  CUdeviceptr d_img_;
  MRow reproj_points_;
  gvdb_params_t params_;

 public:
  GVDBImage(const Eigen::MatrixXf &);
  GVDBImage(const Eigen::MatrixXf &, const gvdb_params_t&);
  void gpuAlloc(const Eigen::MatrixXf &);
  ~GVDBImage() {
    // DEBUG(std::cout<<"[GVDBImage] Deleting object!\n");
    // Free up the memory that we allocated!
    if (d_img_ != 0x0) {
      cuMemFree(d_img_);
    }
  }

  const MRow &get_reproj_points() {
    return reproj_points_;
  }

  const CUdeviceptr &get_buf() {
    return d_img_;
  }
};
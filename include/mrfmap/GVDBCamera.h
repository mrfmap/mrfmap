#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>

#include "gvdb.h"
#include <mrfmap/GVDBCommon.cuh>

// Forward declare classes
class GVDBImage;
class GVDBMRFMap;

typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajorBit> TMat;

void convert_pose_to_opengl_proj_mat(const Eigen::MatrixXf &, TMat &, TMat &);

class GVDBCamera {
 public:
  const uint cam_id_;
  Eigen::MatrixXf pose_;
  Camera3D *cam_;
  bool cum_len_calculated_, is_cached_, needs_update_;

  // Device pointer to this host class
  std::shared_ptr<GVDBMRFMap> map_;
  std::shared_ptr<GVDBImage> img_;

#ifndef FULL_INFERENCE
  // Device pointer to last sent messages index image
  CUdeviceptr d_last_outgoing_img_;
#endif

  GVDBCamera(const Eigen::Matrix4f &pose, const std::shared_ptr<GVDBImage> &img, const std::shared_ptr<GVDBMRFMap> &map, uint cam_id = -1)
      : pose_(pose), img_(img), map_(map), cam_id_(cam_id), cum_len_calculated_(false), is_cached_(false), needs_update_(true) {
    // Allocate camera structures
    gpuAllocate();
    // Call helper method to reproject depth points and activate space in the map
    reprojectAndActivate();
  }

  ~GVDBCamera() {
    if (cam_ != NULL) {
      delete cam_;
    }
    img_.reset();
#ifndef FULL_INFERENCE
    if (d_last_outgoing_img_ != 0x0) {
      cuMemFree(d_last_outgoing_img_);
    }
#endif
  }

  void gpuAllocate();
  void reprojectAndActivate();
  void activateCamera();
  void calculate_cum_len();
};
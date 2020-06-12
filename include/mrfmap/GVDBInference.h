#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
// GVDB library
#include <cuda_runtime_api.h>
// #include <helper_cuda.h>

#include <mrfmap/GVDBCamera.h>
#include <mrfmap/GVDBImage.h>
#include <mrfmap/GVDBMRFMap.h>
#include <mrfmap/GVDBParams.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "gvdb.h"
#include "gvdb_render.h"

class GVDBInference {
 public:
  nvdb::VolumeGVDB* gvdb_;
  std::shared_ptr<GVDBMRFMap> map_;
  std::vector<std::shared_ptr<GVDBCamera>> cameras_;
  std::vector<std::shared_ptr<GVDBImage>> images_;
  CUcontext* ctx_;
#ifdef ENABLE_DEBUG
  CUdeviceptr debug_img_ptr_;
#endif

 private:
  uint cam_id_;
  bool initialized_;

 public:
  GVDBInference(bool create_context = true, bool use_gl = true) : cam_id_(0), initialized_(false) {
    if (create_context) {
      init_context();
    }
    map_ = std::make_shared<GVDBMRFMap>(use_gl);
    gvdb_ = &(map_->gvdb_);
    set_selected_x_y(-1, -1);
    set_selected_voxel(0);
  }

  ~GVDBInference() {
    DEBUG(std::cout << "[GVDBInference] Deleting object!\n");
    cameras_.clear();
    images_.clear();
    map_.reset();

#ifdef ENABLE_DEBUG
    // Deallocate the memory
    cudaCheck(cuMemFree(debug_img_ptr_), "GVDBInference", "perform_inference_ids", "cuMemFree", "", false);
#endif
    // And destroy this context!
    if (*ctx_) {
      cuCtxDestroy(*ctx_);
      delete ctx_;
    }
  }

  void init_context();

  void set_owned_context() {
    // Call to set the CUDA context
    cuCtxSetCurrent(*ctx_);
  }

  void activateCamera(const uint id) {
    if (id < cameras_.size()) {
      cameras_[id]->activateCamera();
    }
  }

  Eigen::MatrixXf& get_cam_pose(const uint id) {
    return cameras_[id]->pose_;
  }

  Eigen::MatrixXf get_occupied_bricks(int lev) {
    return map_->get_occupied_bricks(lev);
  }
  void get_alphas(Eigen::Ref<GVDBImage::MRow> pts) {
    map_->get_alphas(pts);
  }

  void add_camera(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&);
  void add_camera_with_depth(const Eigen::MatrixXf&, const Eigen::MatrixXf&);
  void set_pose(const uint, const Eigen::MatrixXf&);
  void perform_inference();
  void perform_inference_dryrun();
  void perform_inference_ids(const std::vector<uint>&, bool);
  float get_likelihood_image(const uint, Eigen::Ref<GVDBImage::MRow>);
  float get_likelihood_image_at_pose(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&, Eigen::Ref<GVDBImage::MRow>);
  void get_expected_depth_image(const uint, Eigen::Ref<GVDBImage::MRow>);
  float compute_likelihood(const uint);
  void set_selected_x_y(uint, uint);
  void set_selected_voxel(uint64);
  Eigen::MatrixXf get_voxel_coords(std::vector<uint64>&);
  void get_diagnostic_image(uint, Eigen::Ref<GVDBImage::MRow>, uint = 0);
  void get_diagnostic_image_at_pose(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&, Eigen::Ref<GVDBImage::MRow>);
  float get_accuracy_image_at_pose(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&, Eigen::Ref<GVDBImage::MRow>);
};
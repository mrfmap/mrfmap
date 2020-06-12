#pragma once
/**
 * A class to compare likelihoods of different types of maps
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
// GVDB library
#include <cuda_runtime.h>

#include "gvdb.h"
#include "gvdb_render.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <mrfmap/GVDBCommon.cuh>

#include <mrfmap/GVDBCamera.h>
#include <mrfmap/GVDBImage.h>

#include "gvdb.h"

#define OCCUPANCY_CHANNEL 0
#define DIAGNOSTICS_CHANNEL 1

class GVDBMapLikelihoodEstimator {
 private:
  bool use_gl_;

  CUfunction set_channel_kernel_, get_channel_kernel_, render_accuracy_kernel_, render_simple_accuracy_kernel_, render_likelihood_kernel_, render_diagnostics_xy_kernel_, render_occ_thresh_kernel_;

  std::vector<std::shared_ptr<GVDBImage>> images_;
  std::vector<Eigen::MatrixXf> poses_;
  Camera3D* cam_;

 public:
  nvdb::VolumeGVDB* gvdb_;
  CUmodule gvdb_module_;
  CUcontext* ctx_;
  gvdb_params_t params_;  // We don't use the global params here since we might want varying params.

  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatXf;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> PointMats;

  GVDBMapLikelihoodEstimator(gvdb_params_t params, bool use_gl = false) : params_(params), use_gl_(use_gl) {
    std::cout << "Creating context\n";
    init_context();
    gvdb_ = new nvdb::VolumeGVDB();
    gpuAllocate();
    cam_ = new Camera3D;  // Create Camera
    cam_->mNear = 0.1f;
    cam_->mFar = Z_MAX / params_.res;

    set_selected_x_y(-1, -1);
  }

  ~GVDBMapLikelihoodEstimator() {
    DEBUG(std::cout << "[GVDBMapLikelihoodEstimator] Deleting object!\n");
    images_.clear();
    poses_.clear();
    if (cam_ != NULL) {
      delete cam_;
    }
    // Deallocate renderBuf
    if (gvdb_->mRenderBuf[0].gpu != 0x0) {
      cudaCheck(cuMemFree(gvdb_->mRenderBuf[0].gpu), "GVDBMapLikelihoodEstimator", "Destructor", "cuMemFree", "mRenderBuf[0]", false);
    }
    delete gvdb_;
    // And destroy this context!
    if (*ctx_) {
      cuCtxDestroy(*ctx_);
      delete ctx_;
    }
  }

  void init_context() {
    // Create and set context for application
    CUdevice dev;
    int cnt = 0;
    cuInit(0);
    cuDeviceGetCount(&cnt);
    if (cnt == 0) {
      std::cerr << "ERROR:: No CUDA devices found!!! Bailing.\n";
      return;
    }
    // Just use the first device (0)
    cudaCheck(cuDeviceGet(&dev, 0), "GVDBMapLikelihoodEstimator", "init_context", "cuDeviceGet", "", false);
    ctx_ = new CUcontext();
    cudaCheck(cuCtxCreate(ctx_, CU_CTX_SCHED_AUTO, dev), "GVDBMapLikelihoodEstimator", "init_context", "cuCtxCreate", "", false);
    printf("[GVDBMapLikelihoodEstimator] Created context %p \n", (void*)*ctx_);
    // And set this context!
    cudaCheck(cuCtxSetCurrent(NULL), "GVDBMapLikelihoodEstimator", "init_context", "cuCtxSetCurrent", "", false);
    cudaCheck(cuCtxSetCurrent(*ctx_), "GVDBMapLikelihoodEstimator", "init_context", "cuCtxSetCurrent", "", false);
  }

  void push_ctx() {
    cuCtxPushCurrent(gvdb_->getContext());
  }
  void pop_ctx() {
    CUcontext pctx;
    cuCtxPopCurrent(&pctx);
  }

  void gpuLoadModule();

  void gpuAllocate();

  void gpuSetScene();

  Eigen::MatrixXf get_occupied_bricks(int lev);
  void get_alphas(Eigen::Ref<RMatXf> pts);

  void add_camera(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&);
  void set_pose(const uint, const Eigen::MatrixXf&);
  void set_acc_thresh(float);
  void activate_camera(int id);
  void prep_camera_from_pose(Camera3D*, const Eigen::MatrixXf&);
  bool load_map(const PointMats& coords, const RMatXf& occupancies);
  float compute_likelihood(const uint id);
  float get_likelihood_image(const uint cam_id, Eigen::Ref<GVDBImage::MRow> like_img);
  float get_likelihood_image_at_pose(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&, Eigen::Ref<GVDBImage::MRow>);
  float get_likelihood_image_at_pose_with_depth(const Eigen::MatrixXf&, const Eigen::MatrixXf&, Eigen::Ref<GVDBImage::MRow>);
  void get_rendered_image(const uint id, Eigen::Ref<GVDBImage::MRow> eigen_img);
  void get_rendered_image_at_pose(const Eigen::MatrixXf&, Eigen::Ref<GVDBImage::MRow>);
  void get_diagnostic_image(uint id, Eigen::Ref<GVDBImage::MRow> eigen_img);
  void get_diagnostic_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img);
  void get_accuracy_image(uint id, Eigen::Ref<GVDBImage::MRow> eigen_img);
  float get_accuracy_image_at_pose(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&, Eigen::Ref<GVDBImage::MRow>);
  float get_accuracy_image_at_pose_with_depth(const Eigen::MatrixXf&, const Eigen::MatrixXf&, Eigen::Ref<GVDBImage::MRow>);
  float get_simple_accuracy_at_pose(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&);
  float get_simple_accuracy_image_at_pose_with_depth(const Eigen::MatrixXf&, const Eigen::MatrixXf&, Eigen::Ref<GVDBImage::MRow>);
  void get_occ_thresh_image_at_pose(const Eigen::MatrixXf&, const std::shared_ptr<GVDBImage>&, Eigen::Ref<GVDBImage::MRow>, const float);
  void set_selected_x_y(uint x, uint y);
};
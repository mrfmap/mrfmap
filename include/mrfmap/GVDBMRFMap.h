#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>
#include <mrfmap/GVDBParams.h>
#include <mrfmap/GVDBCommon.cuh>
#include "gvdb.h"

class GVDBMRFMap {
 private:
  bool use_gl_;

 public:
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatXf;
  typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> PointMats;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RMatXi;
  nvdb::VolumeGVDB gvdb_;
  CUfunction
      compute_gvdb_message_to_factors_,
      reset_empty_voxels_,
      get_channel_kernel_,
      get_occupied_voxel_mask_kernel_,
      render_accuracy_kernel_,
      render_factors_to_node_kernel_,
      render_length_traversed_kernel_,
      render_likelihood_kernel_,
      render_cumulative_length_kernel_,
      render_subtract_cumulative_length_kernel_,
      render_expected_depth_kernel_,
      render_diagnostics_xy_kernel_;
  CUmodule gvdb_module_;

  // Default constructor
  GVDBMRFMap(bool use_gl = true) : use_gl_(use_gl) {
    gpuAllocate();
  }
  ~GVDBMRFMap() {
    DEBUG(std::cout<<"[GVDBMRFMap] Deleting object!\n");
    push_ctx();
    // Deallocate renderBuf
    if (gvdb_.mRenderBuf[0].gpu != 0x0) {
      cudaCheck(cuMemFree(gvdb_.mRenderBuf[0].gpu), "GVDBMRFMap", "Destructor", "cuMemFree", "mRenderBuf[0]", false);
    }
    pop_ctx();
  }

  void push_ctx() {
    cuCtxPushCurrent(gvdb_.getContext());
  }
  void pop_ctx() {
    CUcontext pctx;
    cuCtxPopCurrent(&pctx);
  }

  void gpuLoadModule();
  void gpuAllocate();
  void gpuSetScene();
  void activateSpace(const PointMats &points);
  Eigen::MatrixXf get_occupied_bricks(int lev);
  void get_occupied_voxels_mask(Eigen::Ref<RMatXi> voxel_mask, float threshold);
  void get_alphas(Eigen::Ref<RMatXf> pts);
};
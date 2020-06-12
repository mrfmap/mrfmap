#include <codetimer/codetimer.h>
#include <cudaProfiler.h>
#include <mrfmap/GVDBInference.h>

#include <chrono>

extern gvdb_params_t gvdb_params_;

// These method is defined in gvdbhelper.cu
float sum_likelihood(CUdeviceptr& ptr, uint num_pixels);
float compute_accuracy(CUdeviceptr& ptr, uint num_pixels);

std::string to_zero_lead(const int value, const unsigned precision)
{
     std::ostringstream oss;
     oss << std::setw(precision) << std::setfill('0') << value;
     return oss.str();
}

void GVDBInference::init_context() {
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
  cudaCheck(cuDeviceGet(&dev, 0), "GVDBInference", "init_context", "cuDeviceGet", "", false);
  ctx_ = new CUcontext();
  cudaCheck(cuCtxCreate(ctx_, CU_CTX_SCHED_AUTO, dev), "GVDBInference", "init_context", "cuCtxCreate", "", false);
  printf("[GVDBInference] Created context %p \n", (void*)*ctx_);
  // And set this context!
  cudaCheck(cuCtxSetCurrent(NULL), "GVDBInference", "init_context", "cuCtxSetCurrent", "", false);
  cudaCheck(cuCtxSetCurrent(*ctx_), "GVDBInference", "init_context", "cuCtxSetCurrent", "", false);

#ifdef ENABLE_DEBUG
  // HACK - Temporarily assign to gvdb's outbuf member of scnInfo
  // First allocate memory...
  cudaCheck(cuMemAlloc(&debug_img_ptr_, gvdb_params_.rows * gvdb_params_.cols * sizeof(float)), "GVDBInference",
            "perform_inference_ids", "cuMemAllocDebug", "", false);
  cudaCheck(cuMemsetD8(debug_img_ptr_, 0, gvdb_params_.rows * gvdb_params_.cols * sizeof(float)), "GVDBInference",
            "perform_inference_ids", "cuMemsetDebug", "", false);
#endif
}

void GVDBInference::add_camera(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr) {
  std::shared_ptr<GVDBCamera> cam = std::make_shared<GVDBCamera>(pose, img_ptr, map_, cameras_.size());
#ifdef ENABLE_DEBUG
  ScnInfo* scn = reinterpret_cast<ScnInfo*>(gvdb_->getScnInfo());
  scn->outbuf = debug_img_ptr_;
#endif
  cameras_.push_back(cam);
}

void GVDBInference::add_camera_with_depth(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& img) {
  map_->push_ctx();
  // Create a GVDBImage; this should just hang around till the camera object is deleted.
  std::shared_ptr<GVDBImage> gvdb_img = std::make_shared<GVDBImage>(img);
  add_camera(pose, gvdb_img);
  map_->pop_ctx();
}

void GVDBInference::set_pose(const uint id, const Eigen::MatrixXf& pose) {
  // First, remove the contribution of all the existing rays
  // a. From cumulative length
  map_->push_ctx();
  gvdb_->SetModule(map_->gvdb_module_);
  gvdb_->RenderKernelAtCachedScene(map_->render_subtract_cumulative_length_kernel_, CUM_LENS, 0, id);
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "set_pose", "cuCtxSynchronize", "", false);
  // b. Now set all the voxels that have no ray passing through them to be cleared out...
  gvdb_->ComputeKernel(map_->gvdb_module_, map_->reset_empty_voxels_, ALPHAS + id + 1, false);
  map_->pop_ctx();

  // TODO?
  cameras_[id]->pose_ = pose;
  // Camera pose has changed, we need to reproject and activate again!
  // TODO: Deactivate old volume? Should I run a quick kernel?
  cameras_[id]->reprojectAndActivate();
}

void GVDBInference::perform_inference() {
  std::vector<uint> ids;
  for (int i = 0; i < cameras_.size(); ++i) {
    ids.push_back(i);
  }
  perform_inference_ids(ids, false);
}

void GVDBInference::perform_inference_dryrun() {
  std::vector<uint> ids;
  for (int i = 0; i < cameras_.size(); ++i) {
    ids.push_back(i);
  }
  perform_inference_ids(ids, true);
}

void GVDBInference::perform_inference_ids(const std::vector<uint>& cam_ids, bool dry_run = false) {
  map_->push_ctx();

  gvdb_->SetModule(map_->gvdb_module_);
  half2 zero_incoming_half2 = __floats2half2_rn(0.0f, 0.0f);
  float assigned_val = *(reinterpret_cast<float*>(&zero_incoming_half2));
  cuProfilerStart();
  // Print memory usage
  std::vector<std::string> outlist;
  // DEBUG(gvdb_->MemoryUsage("gvdb", outlist); for (int n = 0; n < outlist.size(); n++) std::cout <<
  // outlist[n].c_str();); Let's just compute all the cumulative lengths now!
#ifndef FULL_INFERENCE
  if (!dry_run) {
    gvdb_->FillChannel(CUM_LENS, Vector4DF(0, 0, 0, 0));
    for (auto id : cam_ids) {
      // if (!cameras_[id]->cum_len_calculated_)
      // UGH BUT WHAT IF NEW VOXELS ARE ACTIVATED IN SUBSEQUENT CAMERAS IN INCREMENTAL?!?
      // Going to just compute the cum_lens per iteration...
      {
        cameras_[id]->activateCamera();
        gvdb_->RenderKernelAtCachedScene(map_->render_cumulative_length_kernel_, CUM_LENS, 0, id);
        // cuCtxSynchronize();
        // cudaDeviceSynchronize();
        cameras_[id]->calculate_cum_len();
      }
    }
  }
#else
  for (auto id : cam_ids) {
    // Clear out all the weighted incoming messages for all cameras before iteration 0
    for (auto id : cam_ids) {
      gvdb_->FillChannel(ALPHAS + id + 1, Vector4DF(assigned_val, 0, 0, 0));
    }
  }
#endif
  // cuCtxSynchronize();
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "perform_inference_ids", "cuCtxSynchronize2", "", false);
  for (int i = 0; i < gvdb_params_.max_iters; ++i) {
    auto start_block1 = std::chrono::high_resolution_clock::now();
#ifndef FULL_INFERENCE
    if (!dry_run) {
      gvdb_->FillChannel(WEIGHTED_INCOMING, Vector4DF(assigned_val, 0, 0, 0));
    }
    for (int id = 0; id < cameras_.size(); ++id) {
      DEBUG(std::cout << "\n===Activating Camera " << id << "===\n");
      cameras_[id]->activateCamera();
      // Send the id*2+2'th buffer to be written to (last_outgoing_img)
      gvdb_->RenderKernelAtCachedScene(map_->render_factors_to_node_kernel_, dry_run ? 1 : 0, id * 2 + 2, id);
      // cuCtxSynchronize();
    }
    // cuCtxSynchronize();
    // Reset activated camera to selected camera id
    cameras_[cam_id_]->activateCamera();
    gvdb_->ComputeKernel(map_->gvdb_module_, map_->compute_gvdb_message_to_factors_, dry_run ? 1 : 0, false);
#else
    for (int id = 0; id < cameras_.size(); ++id) {
      // Since we reuse CUM_LENS, first zero out cum_lens
      gvdb_->FillChannel(CUM_LENS, Vector4DF(0, 0, 0, 0));
      // Now compute CUM_LENS @TODO: Can I just update this inside factors_to_nodes?
      cameras_[id]->activateCamera();
      cudaCheck(cuCtxSynchronize(), "GVDBInference", "perform_inference_ids", "cuCtxSynchronize3", "", false);
      gvdb_->RenderKernelAtCachedScene(map_->render_cumulative_length_kernel_, CUM_LENS, 0, id);
      // Now that we've computed cum_lens, also send outgoing messages
      gvdb_->RenderKernelAtCachedScene(map_->render_factors_to_node_kernel_, ALPHAS + id + 1, 0, id);
      // This should have updated the individual weighted_incoming, now call nodes_to_factors
      gvdb_->ComputeKernel(map_->gvdb_module_, map_->compute_gvdb_message_to_factors_, ALPHAS + id + 1, false);
      // Weighted incoming channels are cleared within the previous kernel already!
    }
#endif
    DEBUG(std::cout << "-----------------Iteration complete-------------------\n");
    CodeTimer::record("inference" + to_zero_lead(cam_ids.size(), 3), start_block1);
    // std::cout<<"updating apron\n";
    // gvdb_->UpdateApron(ALPHAS);
    // std::cout<<"done?\n";
  }
  cuProfilerStop();
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "perform_inference_ids", "cuCtxSynchronize4", "", false);
  map_->pop_ctx();
}

float GVDBInference::compute_likelihood(const uint id) {
  cameras_[id]->activateCamera();
  gvdb_->RenderKernelAtCachedScene(map_->render_likelihood_kernel_, ALPHAS, 0, id);
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "compute_likelihood", "cuCtxSynchronize", "", false);
  float sum = sum_likelihood(gvdb_->mRenderBuf[0].gpu, gvdb_params_.rows * gvdb_params_.cols);
  return sum;
}

float GVDBInference::get_likelihood_image(const uint id, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  float sum = compute_likelihood(id);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  DEBUG(std::cout << "\nSum is " << sum << "\n");
  return sum;
}

float GVDBInference::get_likelihood_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  cam->mNear = 0.1f;
  cam->mFar = Z_MAX / gvdb_params_.res;

  // TODO: Lock cameras_ here...
  DataPtr new_buffer;
  new_buffer.alloc = map_->gvdb_.mPool;
  new_buffer.cpu = 0x0;
  new_buffer.usedNum = 0;
  new_buffer.lastEle = 0;
  new_buffer.garray = 0;
  new_buffer.grsc = 0;
  new_buffer.glid = -1;
  new_buffer.gpu = 0x0;
  map_->gvdb_.mRenderBuf.push_back(new_buffer);

#ifndef FULL_INFERENCE
  // Each camera has two buffers - one for depth, other for last outgoing
  map_->gvdb_.getScene()->SetDepthBuf(cameras_.size() * 2 + 1);
  map_->gvdb_.mRenderBuf[cameras_.size() * 2 + 1].gpu = img_ptr->get_buf();
#else
  map_->gvdb_.getScene()->SetDepthBuf(cameras_.size() + 1);
  map_->gvdb_.mRenderBuf[cameras_.size() + 1].gpu = img_ptr->get_buf();
#endif

  TMat view_mat = TMat::Zero(4, 4), proj_mat = TMat::Zero(4, 4);
  convert_pose_to_opengl_proj_mat(pose, view_mat, proj_mat);

  cam->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
  cam->updateFrustum();
  map_->gvdb_.getScene()->mCamera = cam;

  // Alright, scene is set, go render!
  gvdb_->RenderKernel(map_->render_likelihood_kernel_, ALPHAS, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "get_likelihood_image_at_pose", "cuCtxSynchronize", "", false);
  float sum = sum_likelihood(gvdb_->mRenderBuf[0].gpu, gvdb_params_.rows * gvdb_params_.cols);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "get_likelihood_image_at_pose", "cuCtxSynchronize", "", false);
  // We done with stuff here, remove the buffer
  map_->gvdb_.mRenderBuf.pop_back();
  // And delete the cam object
  delete cam;
  return sum;
}

void GVDBInference::get_expected_depth_image(const uint id, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  cameras_[id]->activateCamera();
  gvdb_->RenderKernelAtCachedScene(map_->render_expected_depth_kernel_, ALPHAS, 0, id);
  cuCtxSynchronize();
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
}

void GVDBInference::set_selected_x_y(uint x, uint y) {
  CUdeviceptr x_ptr, y_ptr;
  cudaCheck(cuModuleGetGlobal(&x_ptr, NULL, map_->gvdb_module_, "g_selected_x_"), "GVDBInference", "set_selected_x_y",
            "cuModuleGetGlobal", "", false);
  cudaCheck(cuModuleGetGlobal(&y_ptr, NULL, map_->gvdb_module_, "g_selected_y_"), "GVDBInference", "set_selected_x_y",
            "cuModuleGetGlobal", "", false);
  cudaCheck(cuMemcpyHtoD(x_ptr, &x, sizeof(int)), "GVDBInference", "set_selected_x_y", "cuMemcpyHtoD", "", false);
  cudaCheck(cuMemcpyHtoD(y_ptr, &y, sizeof(int)), "GVDBInference", "set_selected_x_y", "cuMemcpyHtoD", "", false);
  std::cout << "[GVDBInference] setting x and y to " << x << ", " << y << "\n";
}

void GVDBInference::set_selected_voxel(uint64 voxel_id) {
  CUdeviceptr voxel_id_ptr;
  cudaCheck(cuModuleGetGlobal(&voxel_id_ptr, NULL, map_->gvdb_module_, "g_selected_vox_id_"), "GVDBInference",
            "set_selected_voxel", "cuModuleGetGlobal", "", false);
  cudaCheck(cuMemcpyHtoD(voxel_id_ptr, &voxel_id, sizeof(uint64)), "GVDBInference", "set_selected_voxel",
            "cuMemcpyHtoD", "", false);
  std::cout << "[GVDBInference] setting selected voxel id to " << voxel_id << "\n";
}

Eigen::MatrixXf GVDBInference::get_voxel_coords(std::vector<uint64>& indices) {
  Node* node;
  Vector3DF bmin, bmax;
  Eigen::MatrixXf bricks = Eigen::MatrixXf::Zero(indices.size(), 6);
  for (int n = 0; n < indices.size(); ++n) {
    std::cout << "Getting node info for index::" << indices[n] << "\n";
    node = gvdb_->getNode(0, 0, indices[n]);
    if (node == 0x0) {
      std::cout << "!!!!!THIS NODE DOES NOT EXIST! GASP!\n";
    } else {
      bmin = gvdb_->getWorldMin(node);
      bmax = gvdb_->getWorldMax(node);
      bricks.row(n) << bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z;
    }
  }
  return bricks;
}

void GVDBInference::get_diagnostic_image(uint id, Eigen::Ref<GVDBImage::MRow> eigen_img, uint mode) {
  // Call the raycast kernel that populates the outBuffer with diagnostic data
  cameras_[id]->activateCamera();
  // HACK - Temporarily assign to gvdb's outbuf member of scnInfo
  // First allocate memory...
  ScnInfo* scn = reinterpret_cast<ScnInfo*>(gvdb_->getScnInfo());
  CUdeviceptr ptr;
  cudaCheck(cuMemAlloc(&ptr, scn->height * scn->width * sizeof(float)), "GVDBInference", "get_diagnostic_image",
            "cuMemAlloc", "", false);
  cudaCheck(cuMemsetD8(ptr, 0, scn->height * scn->width * sizeof(float)), "GVDBInference", "get_diagnostic_image",
            "cuMemAlloc", "", false);
  CUdeviceptr scn_backup = scn->outbuf;
  scn->outbuf = ptr;
  if (mode == 0) {
    gvdb_->RenderKernel(map_->render_diagnostics_xy_kernel_, ALPHAS, 0);
  } else if (mode == 1) {
// Call the render factors kernel with debug flag (chan = 1)
#ifndef FULL_INFERENCE
    gvdb_->RenderKernel(map_->render_factors_to_node_kernel_, 1, id * 2 + 2);
#else
#endif
    // gvdb_->RenderKernelAtCachedScene(map_->render_factors_to_node_kernel_, 1, id * 2 + 2, id);
  }
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "get_diagnostic_image", "cuCtxSynchronize", "", false);
  // Copy the image buffer down to eigen_img
  char* cpu_ptr = reinterpret_cast<char*>(eigen_img.data());
  cudaCheck(cuMemcpyDtoH(cpu_ptr, ptr, scn->height * scn->width * sizeof(float)), "GVDBInference",
            "get_diagnostic_image", "cuMemCpy", "", false);
  // Deallocate the memory
  cudaCheck(cuMemFree(ptr), "GVDBInference", "get_diagnostic_image", "cuMemFree", "", false);
  scn->outbuf = scn_backup;
}

void GVDBInference::get_diagnostic_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  cam->mNear = 0.1f;
  cam->mFar = Z_MAX / gvdb_params_.res;

  // TODO: Lock cameras_ here...
  DataPtr new_buffer;
  new_buffer.alloc = map_->gvdb_.mPool;
  new_buffer.cpu = 0x0;
  new_buffer.usedNum = 0;
  new_buffer.lastEle = 0;
  new_buffer.garray = 0;
  new_buffer.grsc = 0;
  new_buffer.glid = -1;
  new_buffer.gpu = 0x0;
  map_->gvdb_.mRenderBuf.push_back(new_buffer);
#ifndef FULL_INFERENCE
  // Each camera has two buffers - one for depth, other for last outgoing
  map_->gvdb_.getScene()->SetDepthBuf(cameras_.size() * 2 + 1);
  map_->gvdb_.mRenderBuf[cameras_.size() * 2 + 1].gpu = img_ptr->get_buf();
#else
  map_->gvdb_.getScene()->SetDepthBuf(cameras_.size() + 1);
  map_->gvdb_.mRenderBuf[cameras_.size() + 1].gpu = img_ptr->get_buf();
#endif

  TMat view_mat = TMat::Zero(4, 4), proj_mat = TMat::Zero(4, 4);
  convert_pose_to_opengl_proj_mat(pose, view_mat, proj_mat);

  cam->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
  cam->updateFrustum();
  map_->gvdb_.getScene()->mCamera = cam;

  // Alright, scene is set, go render!
  // HACK - Temporarily assign to gvdb's outbuf member of scnInfo
  // First allocate memory...
  cuCtxPushCurrent(gvdb_->getContext());
  ScnInfo* scn = reinterpret_cast<ScnInfo*>(gvdb_->getScnInfo());
  CUdeviceptr ptr;
  cudaCheck(cuMemAlloc(&ptr, scn->height * scn->width * sizeof(float)), "GVDBInference", "get_diagnostic_image_at_pose",
            "cuMemAlloc", "", false);
  cudaCheck(cuMemsetD8(ptr, 0, scn->height * scn->width * sizeof(float)), "GVDBInference", "get_diagnostic_image_at_pose",
            "cuMemsetD8", "", false);
  scn->outbuf = ptr;
  CUcontext pctx;
  cuCtxPopCurrent(&pctx);

  // Call kernel!
  gvdb_->RenderKernel(map_->render_diagnostics_xy_kernel_, ALPHAS, 0);

  cudaCheck(cuCtxSynchronize(), "GVDBInference", "get_diagnostic_image_at_pose", "cuCtxSynchronize", "", false);
  // Copy the image buffer down to eigen_img
  char* cpu_ptr = reinterpret_cast<char*>(eigen_img.data());
  cudaCheck(cuMemcpyDtoH(cpu_ptr, ptr, scn->height * scn->width * sizeof(float)), "GVDBInference",
            "get_diagnostic_image_at_pose", "cuMemCpy", "", false);
  // Deallocate the memory
  cudaCheck(cuMemFree(ptr), "GVDBInference", "get_diagnostic_image_at_pose", "cuMemFree", "", false);
  scn->outbuf = -1;

  // We done with stuff here, remove the buffer
  map_->gvdb_.mRenderBuf.pop_back();
  // And delete the cam object
  delete cam;
}

float GVDBInference::get_accuracy_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  cam->mNear = 0.1f;
  cam->mFar = Z_MAX / gvdb_params_.res;

  // TODO: Lock cameras_ here...
  DataPtr new_buffer;
  new_buffer.alloc = map_->gvdb_.mPool;
  new_buffer.cpu = 0x0;
  new_buffer.usedNum = 0;
  new_buffer.lastEle = 0;
  new_buffer.garray = 0;
  new_buffer.grsc = 0;
  new_buffer.glid = -1;
  new_buffer.gpu = 0x0;
  map_->gvdb_.mRenderBuf.push_back(new_buffer);
#ifndef FULL_INFERENCE
  // Each camera has two buffers - one for depth, other for last outgoing
  map_->gvdb_.getScene()->SetDepthBuf(cameras_.size() * 2 + 1);
  map_->gvdb_.mRenderBuf[cameras_.size() * 2 + 1].gpu = img_ptr->get_buf();
#else
  map_->gvdb_.getScene()->SetDepthBuf(cameras_.size() + 1);
  map_->gvdb_.mRenderBuf[cameras_.size() + 1].gpu = img_ptr->get_buf();
#endif
  TMat view_mat = TMat::Zero(4, 4), proj_mat = TMat::Zero(4, 4);
  convert_pose_to_opengl_proj_mat(pose, view_mat, proj_mat);

  cam->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
  cam->updateFrustum();
  map_->gvdb_.getScene()->mCamera = cam;

  // Alright, scene is set, go render!
  gvdb_->RenderKernel(map_->render_accuracy_kernel_, ALPHAS, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "get_accuracy_image_at_pose", "cuCtxSynchronize", "", false);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  cudaCheck(cuCtxSynchronize(), "GVDBInference", "get_accuracy_image_at_pose", "cuCtxSynchronize", "", false);
  // And now compute the accuracy
  float acc = compute_accuracy(gvdb_->mRenderBuf[0].gpu, gvdb_params_.rows * gvdb_params_.cols);
  // We done with stuff here, remove the buffer
  map_->gvdb_.mRenderBuf.pop_back();
  // And delete the cam object
  delete cam;
  return acc;
}
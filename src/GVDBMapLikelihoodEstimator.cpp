#include <mrfmap/GVDBMapLikelihoodEstimator.h>

// This method is defined in gvdbhelper.cu
float sum_likelihood(CUdeviceptr& ptr, uint num_pixels);
float compute_accuracy(CUdeviceptr& ptr, uint num_pixels);
float compute_tp_fp_tn_fn(CUdeviceptr& ptr, uint num_pixels, float& tp, float& fp, float& tn, float& fn);

void GVDBMapLikelihoodEstimator::gpuAllocate() {
  LOG(gvdb_->SetVerbose(true));
  // Assume that the context has already been set.
  gvdb_->SetCudaDevice(GVDB_DEV_EXISTING, *ctx_);
  DEBUG(std::cout << "Loading Modules...\n");
  gpuLoadModule();
  gvdb_->Initialize();
  // TODO: Couple this with visualization option?
  if (use_gl_) {
    gvdb_->StartRasterGL();
  }
  // DEBUG(std::cout << "\nBefore allocating space, we have::\n"; get_mem_usage(););

  // Create channel
  gvdb_->Configure(params_.gvdb_map_config[0], params_.gvdb_map_config[1], params_.gvdb_map_config[2], params_.gvdb_map_config[3], params_.gvdb_map_config[4]); 
  gvdb_->DestroyChannels();
  // Sets the default allocation to 16x16x1 bricks
  gvdb_->SetChannelDefault(16, 16, 1);

  // Add the default render buffer
  gvdb_->AddRenderBuf(0, params_.cols, params_.rows, sizeof(float));

  DEBUG(std::cout << "Configuring Scene...\n";);
  gpuSetScene();

  DEBUG(std::cout << "Allocating Channel...\n";);
  gvdb_->AddChannel(OCCUPANCY_CHANNEL, T_FLOAT, 1, F_POINT, F_CLAMP);
  // cudaDeviceSynchronize();

  // Set global params
  float logodds_prior = logf(params_.prior / (1.0f - params_.prior));
  CUdeviceptr res_ptr, sigma_ptr, occ_thresh_ptr, acc_thresh_ptr, prior_ptr, logodds_prior_ptr, sigma_lookup_ptr, bias_lookup_ptr, lookup_u_ptr, lookup_v_ptr, lookup_n_ptr;
  cuModuleGetGlobal(&res_ptr, NULL, gvdb_module_, "g_res_");
  cuModuleGetGlobal(&occ_thresh_ptr, NULL, gvdb_module_, "g_occ_thresh_");
  cuModuleGetGlobal(&acc_thresh_ptr, NULL, gvdb_module_, "g_acc_thresh_");
  cuModuleGetGlobal(&prior_ptr, NULL, gvdb_module_, "g_prob_prior_");
  cuModuleGetGlobal(&logodds_prior_ptr, NULL, gvdb_module_, "g_logodds_prob_prior_");
  cuModuleGetGlobal(&sigma_lookup_ptr, NULL, gvdb_module_, "g_depth_sigma_lookup_");
  cuModuleGetGlobal(&bias_lookup_ptr, NULL, gvdb_module_, "g_depth_bias_lookup_");
  cuModuleGetGlobal(&lookup_u_ptr, NULL, gvdb_module_, "g_lookup_u_");
  cuModuleGetGlobal(&lookup_v_ptr, NULL, gvdb_module_, "g_lookup_v_");
  cuModuleGetGlobal(&lookup_n_ptr, NULL, gvdb_module_, "g_lookup_n_");
  cuMemcpyHtoD(res_ptr, &params_.res, sizeof(float));

  cuMemcpyHtoD(occ_thresh_ptr, &params_.occ_thresh, sizeof(float));
  cuMemcpyHtoD(acc_thresh_ptr, &params_.acc_thresh, sizeof(float));
  std::cout << " Accuracy threshold is " << params_.acc_thresh << "\n";
  cuMemcpyHtoD(prior_ptr, &params_.prior, sizeof(float));
  cuMemcpyHtoD(logodds_prior_ptr, &logodds_prior, sizeof(float));

  int n_us = static_cast<int>(ceilf(1.0f * params_.cols / LOOKUP_PX));
  int n_vs = static_cast<int>(ceilf(1.0f * params_.rows / LOOKUP_PX));
  cuMemcpyHtoD(lookup_u_ptr, &n_us, sizeof(int));
  cuMemcpyHtoD(lookup_v_ptr, &n_vs, sizeof(int));
  cuMemcpyHtoD(lookup_n_ptr, &params_.lookup_n, sizeof(int));
  cuMemcpyHtoD(sigma_lookup_ptr, params_.depth_sigma_lookup.data(), params_.depth_sigma_lookup.size() * sizeof(float));
  cuMemcpyHtoD(bias_lookup_ptr, params_.depth_bias_lookup.data(), params_.depth_bias_lookup.size() * sizeof(float));
  // Allocate memory for depth buffer
  gvdb_->mRenderBuf.resize(2);  // Since the first channel is the render channel
  gvdb_->mRenderBuf[1].alloc = gvdb_->mPool;
  gvdb_->mRenderBuf[1].cpu = 0x0;  // no cpu residence yet
  gvdb_->mRenderBuf[1].usedNum = 0;
  gvdb_->mRenderBuf[1].lastEle = 0;
  gvdb_->mRenderBuf[1].garray = 0;
  gvdb_->mRenderBuf[1].grsc = 0;
  gvdb_->mRenderBuf[1].glid = -1;
  gvdb_->mRenderBuf[1].gpu = 0x0;

  // cudaDeviceSynchronize();
}

void GVDBMapLikelihoodEstimator::gpuSetScene() {
  // Scene settings
  // Set volume params
  gvdb_->getScene()->SetSteps(.2f, 16, .2f);            // Set raycasting steps
  gvdb_->getScene()->SetExtinct(-1.0f, 1.0f, 0.0f);     // Set volume extinction
  gvdb_->getScene()->SetVolumeRange(0.05f, 0.0f, 1.f);  // Set volume value range
  gvdb_->getScene()->SetCutoff(0.001f, 0.001f, 0.0f);
  gvdb_->getScene()->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0);
  // Colourmap
  gvdb_->getScene()->LinearTransferFunc(0.00f, 0.25f, Vector4DF(1, 1, 1, 0), Vector4DF(1, 0, 0, 0.1f));
  gvdb_->getScene()->LinearTransferFunc(0.25f, 0.5f, Vector4DF(1, 0, 0, 0.1f), Vector4DF(1, 0, 0, 0.4f));
  gvdb_->getScene()->LinearTransferFunc(0.5f, 0.75f, Vector4DF(1, 0, 0, 0.4f), Vector4DF(1, 1, 0, 0.6f));
  gvdb_->getScene()->LinearTransferFunc(0.75f, 1.0f, Vector4DF(1, 1, 0, 0.6f), Vector4DF(0, 0, 1, 0.8f));
  gvdb_->CommitTransferFunc();
  // Create Light
  Light* lgt = new Light;
  lgt->setOrbit(Vector3DF(299, 57.3f, 0), Vector3DF(132, -20, 50), 20 / params_.res, 100.0);
  gvdb_->getScene()->SetLight(0, lgt);
}

void GVDBMapLikelihoodEstimator::gpuLoadModule() {
  cuModuleLoad(&gvdb_module_, GVDBKERNELS_PTX);
  cuModuleGetFunction(&set_channel_kernel_, gvdb_module_, "set_channel_kernel");
  cuModuleGetFunction(&get_channel_kernel_, gvdb_module_, "get_channel_kernel");
  cuModuleGetFunction(&render_likelihood_kernel_, gvdb_module_, "render_likelihood_kernel");
  cuModuleGetFunction(&render_accuracy_kernel_, gvdb_module_, "render_accuracy_kernel");
  cuModuleGetFunction(&render_simple_accuracy_kernel_, gvdb_module_, "render_simple_accuracy_kernel");
  cuModuleGetFunction(&render_occ_thresh_kernel_, gvdb_module_, "render_occ_thresh_kernel");
  cuModuleGetFunction(&render_diagnostics_xy_kernel_, gvdb_module_, "render_diagnostics_xy_kernel");
}

void GVDBMapLikelihoodEstimator::add_camera(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr) {
  images_.push_back(img_ptr);
  poses_.push_back(pose);
}

void GVDBMapLikelihoodEstimator::set_pose(const uint id, const Eigen::MatrixXf& pose) {
  poses_[id] = pose;
}

void GVDBMapLikelihoodEstimator::set_acc_thresh(float acc_thresh) {
  push_ctx();
  CUdeviceptr acc_thresh_ptr;
  cuModuleGetGlobal(&acc_thresh_ptr, NULL, gvdb_module_, "g_acc_thresh_");
  cuMemcpyHtoD(acc_thresh_ptr, &acc_thresh, sizeof(float));
  pop_ctx();
}

void GVDBMapLikelihoodEstimator::activate_camera(int id) {
  // Copy over image buffer address to this map
  gvdb_->mRenderBuf[1].gpu = images_[id]->get_buf();

  //Set depth buffer
  gvdb_->getScene()->SetDepthBuf(1);

  Eigen::Matrix4f opengl_in_cam;
  opengl_in_cam << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
  Eigen::Matrix4f opengl_in_world = poses_[id] * opengl_in_cam;

  typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajorBit> TMat;
  // Scale metric by resolution
  opengl_in_world.col(3).head<3>() /= params_.res;
  // Offset to center of volume due to requirements of gvdb to only have positive coordinates
  Eigen::Vector3f scaled_map_center_coords = params_.dims * 0.5f / params_.res;
  //Don't shift z coordinate
  scaled_map_center_coords[2] = 0.0f;
  opengl_in_world.col(3).head<3>() += scaled_map_center_coords;
  TMat view_mat = opengl_in_world.inverse();

  // From http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
  // we have
  TMat proj_mat = TMat::Zero();
  proj_mat(0, 0) = params_.K(0, 0) / params_.K(0, 2);
  proj_mat(1, 1) = params_.K(1, 1) / params_.K(1, 2);
  proj_mat(2, 2) = -(cam_->mFar + cam_->mNear) / (cam_->mFar - cam_->mNear);
  proj_mat(2, 3) = -(2.0f * cam_->mFar * cam_->mNear) / (cam_->mFar - cam_->mNear);
  proj_mat(3, 2) = -1.0f;

  view_mat.transposeInPlace();
  proj_mat.transposeInPlace();

  TMat inv_view_rot = TMat::Identity();
  inv_view_rot.block<3, 3>(0, 0) = view_mat.block<3, 3>(0, 0).transpose();

  cam_->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
  cam_->updateFrustum();
  gvdb_->getScene()->mCamera = cam_;  // SetCamera(cam_);
}

bool GVDBMapLikelihoodEstimator::load_map(const PointMats& points, const RMatXf& occupancies) {
  push_ctx();
  DataPtr load_data_ptr, temp1, temp2;
  DEBUG(printf("Transfer data to GPU.\n"));
  uint num_points = points.rows();
  printf("Num points is %u\n", num_points);
  // Sometimes the size of the map might be too big to add in one go, so split
  uint chunk_threshold = 1024 * 1024, pts_remaining = num_points;
  gvdb_->AllocData(load_data_ptr, chunk_threshold, sizeof(Vector3DF), false);
  for (uint chunk_id = 0; pts_remaining > 0; ++chunk_id) {
    uint pts_to_transfer = pts_remaining >= chunk_threshold ? chunk_threshold : pts_remaining % chunk_threshold;
    DEBUG(printf("Transferring %d points \n", pts_to_transfer));
    cudaCheck(cuMemcpyHtoD(load_data_ptr.gpu, (char*)(points.data() + chunk_id * chunk_threshold * 3), pts_to_transfer * sizeof(Vector3DF)), "GVDBMapLikelihoodEstimator", "load_map", "cuMemcpyHtoD", "load_data_ptr", false);
    gvdb_->SetPoints(load_data_ptr, temp1, temp2);
    DEBUG(printf("Activating region \n"));
    Vector3DF m_origin = Vector3DF(0, 0, 0);
    float m_radius = 1.0f;
    gvdb_->AccumulateTopology(pts_to_transfer, m_radius, m_origin);
    pts_remaining -= pts_to_transfer;
  }

  gvdb_->FinishTopology(true, true);
  gvdb_->UpdateAtlas();
  // Get rid of this data now that the atlas  has been updated
  if (load_data_ptr.gpu != 0x0) {
    cudaCheck(cuMemFree(load_data_ptr.gpu), "GVDBMapLikelihoodEstimator", "load_map", "cuMemFree", "", false);
  }

  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "load_map", "cuCtxSynchronize", "", false);
  // Now that all theses bricks have been initialized, fill them up!
  int leafcnt = gvdb_->mPool->getPoolTotalCnt(0, 0);
  DEBUG(std::cout << "Total bricks then are " << leafcnt << "\n");
  // Go brick-by-brick, copying all the data in one go
  gvdb_->SetModule(gvdb_module_);

  // Send VDB Info (*to user module*)
  gvdb_->PrepareVDB();
  CUdeviceptr cuVDBInfo;
  cudaCheck(cuMemAlloc(&cuVDBInfo, sizeof(VDBInfo)), "GVDBMapLikelihoodEstimator", "load_map", "cuMemAlloc", "cuVDBInfo", false);
  cudaCheck(cuMemcpyHtoD(cuVDBInfo, gvdb_->getVDBInfo(), sizeof(VDBInfo)), "GVDBMapLikelihoodEstimator", "load_map", "cuMemcpyHtoD", "cuVDBInfo", false);

  // Copy over the points and the occupancies to GPU
  DataPtr cuda_pts, cuda_alphas;
  uint num_pts = points.rows();
  // Chunking time again!
  pts_remaining = num_points;
  gvdb_->AllocData(cuda_pts, chunk_threshold, sizeof(Vector3DF), false);
  gvdb_->AllocData(cuda_alphas, chunk_threshold, sizeof(float), false);
  for (uint chunk_id = 0; pts_remaining > 0; ++chunk_id) {
    uint pts_to_transfer = pts_remaining >= chunk_threshold ? chunk_threshold : pts_remaining % chunk_threshold;
    // Because GVDB uses incoming data as char * instead of const-ing it...
    char* pts_buffer = const_cast<char*>(reinterpret_cast<const char*>(points.data() + chunk_id * chunk_threshold * 3));
    char* alphas_buffer = const_cast<char*>(reinterpret_cast<const char*>(occupancies.data() + chunk_id * chunk_threshold));
    cudaCheck(cuMemcpyHtoD(cuda_pts.gpu, pts_buffer, pts_to_transfer * sizeof(Vector3DF)), "GVDBMapLikelihoodEstimator", "load_map", "cuMemcpyHtoD", "pts_buffer", false);
    cudaCheck(cuMemcpyHtoD(cuda_alphas.gpu, alphas_buffer, pts_to_transfer * sizeof(float)), "GVDBMapLikelihoodEstimator", "load_map", "cuMemcpyHtoD", "alphas_buffer", false);
    // Assign the alphas memory offset
    int chan = OCCUPANCY_CHANNEL;
    void* args[5] = {&cuVDBInfo, &pts_to_transfer, &cuda_pts.gpu, &cuda_alphas.gpu, &chan};
    int threads = 8;
    int num_grids = pts_to_transfer / (threads * threads * threads) + 1;
    cudaCheck(cuLaunchKernel(set_channel_kernel_, num_grids, 1, 1, threads, threads, threads, 0, NULL, args, NULL), "get_alphas",
              "set_channel_kernel_", "cuLaunch", "cuLaunchKernel", false);
    pts_remaining -= pts_to_transfer;
  }
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "load_map2", "cuCtxSynchronize", "", false);
  std::cout << "done!\n";
  cudaCheck(cuMemFree(cuVDBInfo), "GVDBMRFMap", "get_alphas", "cuMemFree", "", false);
  cudaCheck(cuMemFree(cuda_pts.gpu), "GVDBMapLikelihoodEstimator", "FreeMemLinear", "cuMemFree", "", false);
  cudaCheck(cuMemFree(cuda_alphas.gpu), "GVDBMapLikelihoodEstimator", "FreeMemLinear", "cuMemFree", "", false);

  pop_ctx();
  return true;
}

void GVDBMapLikelihoodEstimator::prep_camera_from_pose(Camera3D* cam, const Eigen::MatrixXf& pose) {
  cam->mNear = 0.1f;
  cam->mFar = Z_MAX / params_.res;

  TMat view_mat = TMat::Zero(4, 4), proj_mat = TMat::Zero(4, 4);
  convert_pose_to_opengl_proj_mat(pose, view_mat, proj_mat);
  cam->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
  cam->updateFrustum();
  gvdb_->getScene()->mCamera = cam;
}

float GVDBMapLikelihoodEstimator::compute_likelihood(const uint id) {
  activate_camera(id);
  gvdb_->RenderKernel(render_likelihood_kernel_, OCCUPANCY_CHANNEL, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "compute_likelihood", "cuCtxSynchronize", "", false);
  float sum = sum_likelihood(gvdb_->mRenderBuf[0].gpu, params_.rows * params_.cols);
  return sum;
}

void GVDBMapLikelihoodEstimator::get_rendered_image(const uint id, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  std::cout << "rendering image\n";
  gvdb_->Render(SHADE_VOXEL, 0, 0);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
}

float GVDBMapLikelihoodEstimator::get_likelihood_image(const uint id, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  float sum = compute_likelihood(id);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  DEBUG(std::cout << "\nSum is " << sum << "\n");
  return sum;
}

void GVDBMapLikelihoodEstimator::get_rendered_image_at_pose(const Eigen::MatrixXf& pose, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  prep_camera_from_pose(cam, pose);
  gvdb_->getScene()->mCamera = cam;

  std::cout << "rendering image\n";
  gvdb_->Render(SHADE_VOXEL, OCCUPANCY_CHANNEL, 0);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  pop_ctx();
}

float GVDBMapLikelihoodEstimator::get_likelihood_image_at_pose_with_depth(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& depth_img, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();
  std::shared_ptr<GVDBImage> img_ptr = std::make_shared<GVDBImage>(depth_img, params_);
  pop_ctx();
  return get_likelihood_image_at_pose(pose, img_ptr, eigen_img);
}

float GVDBMapLikelihoodEstimator::get_likelihood_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  prep_camera_from_pose(cam, pose);

  gvdb_->getScene()->SetDepthBuf(1);
  gvdb_->mRenderBuf[1].gpu = img_ptr->get_buf();

  // Alright, scene is set, go render!
  gvdb_->RenderKernel(render_likelihood_kernel_, OCCUPANCY_CHANNEL, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_likelihood_image_at_pose", "cuCtxSynchronize", "", false);
  float sum = sum_likelihood(gvdb_->mRenderBuf[0].gpu, params_.rows * params_.cols);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_likelihood_image_at_pose", "cuCtxSynchronize", "", false);

  // And delete the cam object
  delete cam;
  pop_ctx();
  return sum;
}

void GVDBMapLikelihoodEstimator::set_selected_x_y(uint x, uint y) {
  CUdeviceptr x_ptr, y_ptr;
  cudaCheck(cuModuleGetGlobal(&x_ptr, NULL, gvdb_module_, "g_selected_x_"), "GVDBMapLikelihoodEstimator", "set_selected_x_y",
            "cuModuleGetGlobal", "", false);
  cudaCheck(cuModuleGetGlobal(&y_ptr, NULL, gvdb_module_, "g_selected_y_"), "GVDBMapLikelihoodEstimator", "set_selected_x_y",
            "cuModuleGetGlobal", "", false);
  cudaCheck(cuMemcpyHtoD(x_ptr, &x, sizeof(int)), "GVDBMapLikelihoodEstimator", "set_selected_x_y", "cuMemcpyHtoD", "", false);
  cudaCheck(cuMemcpyHtoD(y_ptr, &y, sizeof(int)), "GVDBMapLikelihoodEstimator", "set_selected_x_y", "cuMemcpyHtoD", "", false);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "set_selected_x_y", "cuCtxSynchronize", "", false);
  std::cout << "[GVDBMapLikelihoodEstimator] setting x and y to " << x << ", " << y << "\n";
}

void GVDBMapLikelihoodEstimator::get_accuracy_image(uint id, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  activate_camera(id);
  gvdb_->RenderKernel(render_accuracy_kernel_, OCCUPANCY_CHANNEL, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "compute_likelihood", "cuCtxSynchronize", "", false);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "set_selected_x_y", "cuCtxSynchronize", "", false);
}

float GVDBMapLikelihoodEstimator::get_accuracy_image_at_pose_with_depth(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& depth_img, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();
  std::shared_ptr<GVDBImage> img_ptr = std::make_shared<GVDBImage>(depth_img, params_);
  pop_ctx();
  return get_accuracy_image_at_pose(pose, img_ptr, eigen_img);
}

float GVDBMapLikelihoodEstimator::get_accuracy_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  prep_camera_from_pose(cam, pose);

  gvdb_->getScene()->SetDepthBuf(1);
  gvdb_->mRenderBuf[1].gpu = img_ptr->get_buf();

  // Alright, scene is set, go render!
  gvdb_->RenderKernel(render_accuracy_kernel_, OCCUPANCY_CHANNEL, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_accuracy_image_at_pose", "cuCtxSynchronize", "", false);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_accuracy_image_at_pose", "cuCtxSynchronize", "", false);
  // And now compute the accuracy
  float acc = compute_accuracy(gvdb_->mRenderBuf[0].gpu, params_.rows * params_.cols);
  // And delete the cam object
  delete cam;
  pop_ctx();
  return acc;
}

float GVDBMapLikelihoodEstimator::get_simple_accuracy_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr) {
  push_ctx();
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  prep_camera_from_pose(cam, pose);

  gvdb_->getScene()->SetDepthBuf(1);
  gvdb_->mRenderBuf[1].gpu = img_ptr->get_buf();

  // Alright, scene is set, go render!
  gvdb_->RenderKernel(render_simple_accuracy_kernel_, OCCUPANCY_CHANNEL, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_simple_accuracy_image_at_pose", "cuCtxSynchronize", "", false);
  // And now compute the accuracy
  float tp = 0.f, tn = 0.f, fp = 0.f, fn = 0.f;
  float acc = compute_tp_fp_tn_fn(gvdb_->mRenderBuf[0].gpu, params_.rows * params_.cols, tp, fp, tn, fn);
  // And delete the cam object
  delete cam;
  pop_ctx();
  return acc;
}

float GVDBMapLikelihoodEstimator::get_simple_accuracy_image_at_pose_with_depth(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& depth_img, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();
  std::shared_ptr<GVDBImage> img_ptr = std::make_shared<GVDBImage>(depth_img, params_);
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  prep_camera_from_pose(cam, pose);

  gvdb_->getScene()->SetDepthBuf(1);
  gvdb_->mRenderBuf[1].gpu = img_ptr->get_buf();

  // Alright, scene is set, go render!
  gvdb_->RenderKernel(render_simple_accuracy_kernel_, OCCUPANCY_CHANNEL, 0);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_simple_accuracy_image_at_pose", "cuCtxSynchronize", "", false);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_simple_accuracy_image_at_pose", "cuCtxSynchronize", "", false);
  // And now compute the accuracy
  float tp = 0.f, tn = 0.f, fp = 0.f, fn = 0.f;
  float acc = compute_tp_fp_tn_fn(gvdb_->mRenderBuf[0].gpu, params_.rows * params_.cols, tp, fp, tn, fn);
  // And delete the cam object
  delete cam;
  pop_ctx();
  return acc;
}

void GVDBMapLikelihoodEstimator::get_occ_thresh_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img, const float occ_thresh) {
  // First set the current scene camera to be at this pose
  Camera3D* cam = new Camera3D;
  prep_camera_from_pose(cam, pose);

  gvdb_->getScene()->SetDepthBuf(1);
  gvdb_->mRenderBuf[1].gpu = img_ptr->get_buf();

  // Alright, scene is set, go render!
  // First, set occ_thresh
  CUdeviceptr occ_thresh_ptr;
  cudaCheck(cuModuleGetGlobal(&occ_thresh_ptr, NULL, gvdb_module_, "g_occ_thresh_"), "GVDBMapLikelihoodEstimator", "get_occ_thresh_image_at_pose", "cuModuleGetGlobal", "", false);

  cudaCheck(cuMemcpyHtoD(occ_thresh_ptr, &occ_thresh, sizeof(float)), "GVDBMapLikelihoodEstimator", "get_occ_thresh_image_at_pose", "cuMemcpyHtoD", "", false);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_occ_thresh_image_at_pose", "cuCtxSynchronize", "", false);
  // Ok, now render kernel
  gvdb_->RenderKernel(render_occ_thresh_kernel_, OCCUPANCY_CHANNEL, 0);

  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_occ_thresh_image_at_pose", "cuCtxSynchronize", "", false);
  gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(eigen_img.data()));
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_occ_thresh_image_at_pose", "cuCtxSynchronize", "", false);

  // And delete the cam object
  delete cam;
}

void GVDBMapLikelihoodEstimator::get_diagnostic_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();

  Camera3D* cam = new Camera3D;
  prep_camera_from_pose(cam, pose);

  gvdb_->getScene()->SetDepthBuf(1);
  gvdb_->mRenderBuf[1].gpu = img_ptr->get_buf();

  // Alright, scene is set, go render!
  // Call the raycast kernel that populates the outBuffer with diagnostic data
  // HACK - Temporarily assign to gvdb's outbuf member of scnInfo
  // First allocate memory...
  ScnInfo* scn = reinterpret_cast<ScnInfo*>(gvdb_->getScnInfo());
  CUdeviceptr ptr;
  cudaCheck(cuMemAlloc(&ptr, scn->height * scn->width * sizeof(float)), "GVDBMapLikelihoodEstimator", "get_diagnostic_image",
            "cuMemAlloc", "", false);
  cudaCheck(cuMemsetD8(ptr, 0, scn->height * scn->width * sizeof(float)), "GVDBMapLikelihoodEstimator", "get_diagnostic_image",
            "cuMemAlloc", "", false);
  std::cout << "Scened height and width are " << scn->height << ", " << scn->width << "\n";
  scn->outbuf = ptr;
  printf("Before calling render kernel, pointer is %p", ptr);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_diagnostic_image", "cuCtxSynchronize", "", false);

  gvdb_->RenderKernel(render_diagnostics_xy_kernel_, OCCUPANCY_CHANNEL, 0);

  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_diagnostic_image", "cuCtxSynchronize", "", false);
  // Copy the image buffer down to eigen_img
  char* cpu_ptr = reinterpret_cast<char*>(eigen_img.data());
  cudaCheck(cuMemcpyDtoH(cpu_ptr, ptr, scn->height * scn->width * sizeof(float)), "GVDBMapLikelihoodEstimator",
            "get_diagnostic_image", "cuMemCpy", "", false);
  // Deallocate the memory
  cudaCheck(cuMemFree(ptr), "GVDBMapLikelihoodEstimator", "get_diagnostic_image", "cuMemFree", "", false);
  scn->outbuf = -1;
  pop_ctx();
  delete cam;
}

void GVDBMapLikelihoodEstimator::get_diagnostic_image(uint id, Eigen::Ref<GVDBImage::MRow> eigen_img) {
  push_ctx();
  // Call the raycast kernel that populates the outBuffer with diagnostic data
  activate_camera(id);
  // HACK - Temporarily assign to gvdb's outbuf member of scnInfo
  // First allocate memory...
  ScnInfo* scn = reinterpret_cast<ScnInfo*>(gvdb_->getScnInfo());
  CUdeviceptr ptr;
  cudaCheck(cuMemAlloc(&ptr, scn->height * scn->width * sizeof(float)), "GVDBMapLikelihoodEstimator", "get_diagnostic_image",
            "cuMemAlloc", "", false);
  cudaCheck(cuMemsetD8(ptr, 0, scn->height * scn->width * sizeof(float)), "GVDBMapLikelihoodEstimator", "get_diagnostic_image",
            "cuMemAlloc", "", false);
  std::cout << "Scened height and width are " << scn->height << ", " << scn->width << "\n";
  scn->outbuf = ptr;
  printf("Before calling render kernel, pointer is %p", ptr);
  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_diagnostic_image", "cuCtxSynchronize", "", false);

  gvdb_->RenderKernel(render_diagnostics_xy_kernel_, OCCUPANCY_CHANNEL, 0);

  cudaCheck(cuCtxSynchronize(), "GVDBMapLikelihoodEstimator", "get_diagnostic_image", "cuCtxSynchronize", "", false);
  // Copy the image buffer down to eigen_img
  char* cpu_ptr = reinterpret_cast<char*>(eigen_img.data());
  cudaCheck(cuMemcpyDtoH(cpu_ptr, ptr, scn->height * scn->width * sizeof(float)), "GVDBMapLikelihoodEstimator",
            "get_diagnostic_image", "cuMemCpy", "", false);
  // Deallocate the memory
  cudaCheck(cuMemFree(ptr), "GVDBMapLikelihoodEstimator", "get_diagnostic_image", "cuMemFree", "", false);
  scn->outbuf = -1;
  pop_ctx();
}

Eigen::MatrixXf GVDBMapLikelihoodEstimator::get_occupied_bricks(int lev) {
  Node* node;
  Vector3DF bmin, bmax;
  int node_cnt = gvdb_->getNumNodes(lev);
  Eigen::MatrixXf brick_centers = Eigen::MatrixXf::Zero(node_cnt, 3);

  // Vector3DF worldmin = gvdb_->getWorldMin();
  // printf("\nWorldmin is %f %f %f\n", worldmin.x, worldmin.y, worldmin.z);
  for (int n = 0; n < node_cnt; n++) {  // draw all nodes at this level
    node = gvdb_->getNodeAtLevel(n, lev);
    bmin = gvdb_->getWorldMin(node);
    bmax = gvdb_->getWorldMax(node);
    brick_centers.row(n) << (bmax.x + bmin.x) / 2.0f, (bmax.y + bmin.y) / 2.0f, (bmax.z + bmin.z) / 2.0f;
  }
  return brick_centers;
}

void GVDBMapLikelihoodEstimator::get_alphas(Eigen::Ref<RMatXf> pts) {
  push_ctx();
  // Go brick-by-brick, copying all the data in one go
  gvdb_->SetModule(gvdb_module_);

  // Send VDB Info (*to user module*)
  gvdb_->PrepareVDB();
  CUdeviceptr cuVDBInfo;
  cudaCheck(cuMemAlloc(&cuVDBInfo, sizeof(VDBInfo)), "GVDBMapLikelihood", "get_alphas", "cuMemAlloc", "cuVDBInfo", false);
  cudaCheck(cuMemcpyHtoD(cuVDBInfo, gvdb_->getVDBInfo(), sizeof(VDBInfo)), "GVDBMapLikelihood", "get_alphas", "cuMemcpyHtoD",
            "cuVDBInfo", false);

  // Get number of bricks
  Node* node;
  Vector3DI axisres = gvdb_->mPool->getAtlasRes(0);
  Vector3DI pos;
  int3 res = make_int3(gvdb_->getRes(0), gvdb_->getRes(0), gvdb_->getRes(0));
  int sz = gvdb_->getVoxCnt(0) * sizeof(float);  // res*res*res*sizeof(float)
  int leafcnt = gvdb_->mPool->getPoolTotalCnt(0, 0);
  // pts.resize(leafcnt, gvdb_->getVoxCnt(0));  // to store x,y,z, and value
  if (pts.rows() != leafcnt && pts.cols() != gvdb_->getVoxCnt(0)) {
    throw std::runtime_error("Dimensions for the alphas array are incorrect!");
    return;
  }
  DataPtr p;
  // gvdb_->mPool->CreateMemLinear ( p, 0x0, 1, sz, true );
  p.alloc = gvdb_->mPool;
  p.lastEle = sz;
  p.usedNum = sz;
  p.max = sz;
  p.stride = 1;
  p.size = (uint64)sz * (uint64)1;
  p.subdim = Vector3DI(0, 0, 0);
  // Allocate memory for the gpu equiv that the data will be transferred to
  cudaCheck(cuMemAlloc(&p.gpu, p.size), "GVDBMapLikelihood", "get_alphas", "cuMemAlloc", "", false);

  for (int n = 0; n < leafcnt; n++) {
    node = gvdb_->getNode(0, 0, n);
    pos = node->mPos;

    // Assign the alphas memory offset
    p.cpu = reinterpret_cast<char*>(pts.data()) + n * pts.rowStride() * sizeof(float);
    int chan = OCCUPANCY_CHANNEL;
    // gvdb_->mPool->AtlasRetrieveTexXYZ(ALPHAS, node->mValue, p);
    void* args[5] = {&cuVDBInfo, &node->mValue, &res, &p.gpu, &chan};
    cudaCheck(cuLaunchKernel(get_channel_kernel_, 1, 1, 1, res.x, res.y, res.z, 0, NULL, args, NULL), "get_alphas",
              "get_channel_kernel_", "cuLaunch", "cuLaunchKernel", false);
    gvdb_->mPool->RetrieveMem(p);
    // cuCtxSynchronize();
  }
  // Free the gpu memory
  cudaCheck(cuMemFree(p.gpu), "GVDBMapLikelihood", "get_alphas", "cuMemFree", "", false);
  cudaCheck(cuMemFree(cuVDBInfo), "GVDBMapLikelihood", "get_alphas", "cuMemFree", "", false);
  pop_ctx();
}
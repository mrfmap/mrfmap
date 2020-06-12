#include <mrfmap/GVDBMRFMap.h>

extern gvdb_params_t gvdb_params_;

void GVDBMRFMap::gpuLoadModule() {
  cuModuleLoad(&gvdb_module_, GVDBKERNELS_PTX);
  cuModuleGetFunction(&render_accuracy_kernel_, gvdb_module_, "render_accuracy_kernel");
  cuModuleGetFunction(&render_factors_to_node_kernel_, gvdb_module_, "render_factors_to_node_kernel");
  cuModuleGetFunction(&render_length_traversed_kernel_, gvdb_module_, "render_length_traversed_kernel");
  cuModuleGetFunction(&render_likelihood_kernel_, gvdb_module_, "render_likelihood_kernel");
  cuModuleGetFunction(&render_cumulative_length_kernel_, gvdb_module_, "render_cumulative_length_kernel");
  cuModuleGetFunction(&render_subtract_cumulative_length_kernel_, gvdb_module_, "render_subtract_cumulative_length_kernel");

  cuModuleGetFunction(&render_expected_depth_kernel_, gvdb_module_, "render_expected_depth_kernel");
  cuModuleGetFunction(&render_diagnostics_xy_kernel_, gvdb_module_, "render_diagnostics_xy_kernel");

  cuModuleGetFunction(&compute_gvdb_message_to_factors_, gvdb_module_, "compute_gvdb_message_to_factors");
  cuModuleGetFunction(&reset_empty_voxels_, gvdb_module_, "reset_empty_voxels");
  cuModuleGetFunction(&get_channel_kernel_, gvdb_module_, "get_channel_kernel");
  cuModuleGetFunction(&get_occupied_voxel_mask_kernel_, gvdb_module_, "get_occupied_voxel_mask_kernel");
}

void GVDBMRFMap::gpuAllocate() {
  LOG(std::cout << "[GVDBMRFMap] Initialized with:\n"
                << gvdb_params_ << "\n";);
  LOG(gvdb_.SetVerbose(true));
  // Assume that the context has already been set.
  gvdb_.SetCudaDevice(GVDB_DEV_CURRENT);
  DEBUG(std::cout << "Loading Modules...\n");
  gpuLoadModule();
  gvdb_.Initialize();
  // TODO: Couple this with visualization option?
  if (use_gl_) {
    gvdb_.StartRasterGL();
  }
  // DEBUG(std::cout << "\nBefore allocating space, we have::\n"; get_mem_usage(););

  // Create channel
  gvdb_.Configure(gvdb_params_.gvdb_map_config[0], gvdb_params_.gvdb_map_config[1], gvdb_params_.gvdb_map_config[2], gvdb_params_.gvdb_map_config[3], gvdb_params_.gvdb_map_config[4]);
  gvdb_.DestroyChannels();
  // Sets the default allocation to 16x16x1 bricks
  gvdb_.SetChannelDefault(16, 16, 1);

  // Add the default render buffer
  gvdb_.AddRenderBuf(0, gvdb_params_.cols, gvdb_params_.rows, sizeof(float));

  DEBUG(std::cout << "Configuring Scene...\n";);
  gpuSetScene();

  DEBUG(std::cout << "Allocating Channels...\n";);
  float pos_log_msg_prior = log(gvdb_params_.prior) - log(1.0f - gvdb_params_.prior);
  half2 pos_log_msg_prior_half2 = __floats2half2_rn(0.0f, pos_log_msg_prior);
  float assigned_val = *(reinterpret_cast<float *>(&pos_log_msg_prior_half2));

  Vector4DF val = {assigned_val, 0.0f, 0.0f, 0.0f};
  gvdb_.AddChannel(POS_LOG_MSG_SUM, T_FLOAT, 1, F_POINT, F_CLAMP, Vector3DI(0, 0, 0), true, val);
  DEBUG(std::cout << " Added POS_LOG_MSG_SUM\n";);
  gvdb_.AddChannel(CUM_LENS, T_FLOAT, 1, F_POINT, F_CLAMP, Vector3DI(0, 0, 0), false);
  DEBUG(std::cout << " Added CUM_LENS\n";);
#ifndef FULL_INFERENCE
  gvdb_.AddChannel(WEIGHTED_INCOMING, T_FLOAT, 1, F_LINEAR, F_CLAMP, Vector3DI(0, 0, 0), false);
  DEBUG(std::cout << " Added WEIGHTED_INCOMING\n";);
#endif
  gvdb_.AddChannel(ALPHAS, T_FLOAT, 1, F_POINT, F_CLAMP);
  DEBUG(std::cout << " Added ALPHAS\n";);
  // cudaDeviceSynchronize();

  // Set global params
  CUdeviceptr res_ptr, sigma_ptr, occ_thresh_ptr, prior_ptr, logodds_prior_ptr, sigma_lookup_ptr, bias_lookup_ptr, lookup_u_ptr, lookup_v_ptr, lookup_n_ptr;
  cuModuleGetGlobal(&res_ptr, NULL, gvdb_module_, "g_res_");
  cuModuleGetGlobal(&occ_thresh_ptr, NULL, gvdb_module_, "g_occ_thresh_");
  cuModuleGetGlobal(&prior_ptr, NULL, gvdb_module_, "g_prob_prior_");
  cuModuleGetGlobal(&logodds_prior_ptr, NULL, gvdb_module_, "g_logodds_prob_prior_");
  cuModuleGetGlobal(&sigma_lookup_ptr, NULL, gvdb_module_, "g_depth_sigma_lookup_");
  cuModuleGetGlobal(&bias_lookup_ptr, NULL, gvdb_module_, "g_depth_bias_lookup_");
  cuModuleGetGlobal(&lookup_u_ptr, NULL, gvdb_module_, "g_lookup_u_");
  cuModuleGetGlobal(&lookup_v_ptr, NULL, gvdb_module_, "g_lookup_v_");
  cuModuleGetGlobal(&lookup_n_ptr, NULL, gvdb_module_, "g_lookup_n_");
  cuMemcpyHtoD(res_ptr, &gvdb_params_.res, sizeof(float));

  cuMemcpyHtoD(occ_thresh_ptr, &gvdb_params_.occ_thresh, sizeof(float));
  cuMemcpyHtoD(prior_ptr, &gvdb_params_.prior, sizeof(float));
  cuMemcpyHtoD(logodds_prior_ptr, &pos_log_msg_prior, sizeof(float));

  int n_us = static_cast<int>(ceilf(1.0f * gvdb_params_.cols / LOOKUP_PX));
  int n_vs = static_cast<int>(ceilf(1.0f * gvdb_params_.rows / LOOKUP_PX));
  cuMemcpyHtoD(lookup_u_ptr, &n_us, sizeof(int));
  cuMemcpyHtoD(lookup_v_ptr, &n_vs, sizeof(int));
  cuMemcpyHtoD(lookup_n_ptr, &gvdb_params_.lookup_n, sizeof(int));
  cuMemcpyHtoD(sigma_lookup_ptr, gvdb_params_.depth_sigma_lookup.data(), gvdb_params_.depth_sigma_lookup.size() * sizeof(float));
  cuMemcpyHtoD(bias_lookup_ptr, gvdb_params_.depth_bias_lookup.data(), gvdb_params_.depth_bias_lookup.size() * sizeof(float));
  // cudaDeviceSynchronize();
}

void GVDBMRFMap::gpuSetScene() {
  // Scene settings
  // Set volume params
  gvdb_.getScene()->SetSteps(.2f, 16, .2f);            // Set raycasting steps
  gvdb_.getScene()->SetExtinct(-1.0f, 1.0f, 0.0f);     // Set volume extinction
  gvdb_.getScene()->SetVolumeRange(0.05f, 0.0f, 1.f);  // Set volume value range
  gvdb_.getScene()->SetCutoff(0.005f, 0.01f, 0.0f);
  // gvdb_.getScene()->SetBackgroundClr(1.0f, 1.0f, 1.0f, 1.0);
  gvdb_.getScene()->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0);
  // Colourmap
  gvdb_.getScene()->LinearTransferFunc(0.0f, 0.05f, Vector4DF(1, 1, 1, 0.0f), Vector4DF(1, 0, 0, 0.1f));
  gvdb_.getScene()->LinearTransferFunc(0.1f, 0.25f, Vector4DF(1, 0, 0, 0.1f), Vector4DF(1, 0, 0, 0.4f));
  gvdb_.getScene()->LinearTransferFunc(0.25f, 0.50f, Vector4DF(1, 0, 0, 0.4f), Vector4DF(1, 1, 0, 0.6f));
  gvdb_.getScene()->LinearTransferFunc(0.50f, 0.75f, Vector4DF(1, 1, 0, 0.6f), Vector4DF(0, 0, 1, 0.8f));
  gvdb_.getScene()->LinearTransferFunc(0.75f, 1.00f, Vector4DF(0, 0, 1, 0.8f), Vector4DF(0, 0, 1, 0.9f));
  gvdb_.CommitTransferFunc();
  // Create Light
  Light *lgt = new Light;
  lgt->setOrbit(Vector3DF(299, 57.3f, 0), Vector3DF(132, -20, 50), 20 / gvdb_params_.res, 100.0);
  gvdb_.getScene()->SetLight(0, lgt);
}

void GVDBMRFMap::activateSpace(const PointMats &points) {
  DataPtr imgptr, temp1, temp2;
  DEBUG(printf("Transfer data to GPU.\n"));
  uint num_points = points.rows();
  gvdb_.AllocData(imgptr, num_points, sizeof(Vector3DF), false);
  gvdb_.CommitData(imgptr, num_points, (char *)points.data(), 0, sizeof(Vector3DF));
  gvdb_.SetPoints(imgptr, temp1, temp2);
  DEBUG(printf("Activating region \n"));
  Vector3DF m_origin = Vector3DF(0, 0, 0);
  float m_radius = 1.0f;
  gvdb_.AccumulateTopology(num_points, m_radius, m_origin);
  // TODO: Initialize the new pos_log_msg locations with the prior value???
  gvdb_.FinishTopology(true, true);
  gvdb_.UpdateAtlas();
  // Get rid of this data now that the atlas  has been updated
  if (imgptr.gpu != 0x0) {
    cudaCheck(cuMemFree(imgptr.gpu), "GVDBMRFMap", "activateSpace", "cuMemFree", "", false);
  }
}

Eigen::MatrixXf GVDBMRFMap::get_occupied_bricks(int lev) {
  Node *node;
  Vector3DF bmin, bmax;
  int node_cnt = gvdb_.getNumNodes(lev);
  Eigen::MatrixXf brick_centers = Eigen::MatrixXf::Zero(node_cnt, 3);

  // Vector3DF worldmin = gvdb_.getWorldMin();
  // printf("\nWorldmin is %f %f %f\n", worldmin.x, worldmin.y, worldmin.z);
  for (int n = 0; n < node_cnt; n++) {  // draw all nodes at this level
    node = gvdb_.getNodeAtLevel(n, lev);
    bmin = gvdb_.getWorldMin(node);
    bmax = gvdb_.getWorldMax(node);
    brick_centers.row(n) << (bmax.x + bmin.x) / 2.0f, (bmax.y + bmin.y) / 2.0f, (bmax.z + bmin.z) / 2.0f;
  }
  return brick_centers;
}

void GVDBMRFMap::get_alphas(Eigen::Ref<RMatXf> pts) {
  push_ctx();
  // Go brick-by-brick, copying all the data in one go
  gvdb_.SetModule(gvdb_module_);

  // Send VDB Info (*to user module*)
  gvdb_.PrepareVDB();
  CUdeviceptr cuVDBInfo;
  cudaCheck(cuMemAlloc(&cuVDBInfo, sizeof(VDBInfo)), "GVDBMRFMap", "get_alphas", "cuMemAlloc", "cuVDBInfo", false);
  cudaCheck(cuMemcpyHtoD(cuVDBInfo, gvdb_.getVDBInfo(), sizeof(VDBInfo)), "GVDBMRFMap", "get_alphas", "cuMemcpyHtoD",
            "cuVDBInfo", false);

  // Get number of bricks
  Node *node;
  Vector3DI axisres = gvdb_.mPool->getAtlasRes(0);
  Vector3DI pos;
  int3 res = make_int3(gvdb_.getRes(0), gvdb_.getRes(0), gvdb_.getRes(0));
  int sz = gvdb_.getVoxCnt(0) * sizeof(float);  // res*res*res*sizeof(float)
  int leafcnt = gvdb_.mPool->getPoolTotalCnt(0, 0);
  // pts.resize(leafcnt, gvdb_.getVoxCnt(0));  // to store x,y,z, and value
  if (pts.rows() != leafcnt || pts.cols() != gvdb_.getVoxCnt(0)) {
    throw std::runtime_error("Dimensions for the alphas array are incorrect!");
    return;
  }
  DataPtr p;
  // gvdb_.mPool->CreateMemLinear ( p, 0x0, 1, sz, true );
  p.alloc = gvdb_.mPool;
  p.lastEle = sz;
  p.usedNum = sz;
  p.max = sz;
  p.stride = 1;
  p.size = (uint64)sz * (uint64)1;
  p.subdim = Vector3DI(0, 0, 0);
  // Allocate memory for the gpu equiv that the data will be transferred to
  cudaCheck(cuMemAlloc(&p.gpu, p.size), "GVDBMRFMap", "get_alphas", "cuMemAlloc", "", false);

  for (int n = 0; n < leafcnt; n++) {
    node = gvdb_.getNode(0, 0, n);
    pos = node->mPos;

    // Assign the alphas memory offset
    p.cpu = reinterpret_cast<char *>(pts.data()) + n * pts.rowStride() * sizeof(float);
    int chan = ALPHAS;
    // gvdb_.mPool->AtlasRetrieveTexXYZ(ALPHAS, node->mValue, p);
    void *args[5] = {&cuVDBInfo, &node->mValue, &res, &p.gpu, &chan};
    cudaCheck(cuLaunchKernel(get_channel_kernel_, 1, 1, 1, res.x, res.y, res.z, 0, NULL, args, NULL), "get_alphas",
              "get_channel_kernel_", "cuLaunch", "cuLaunchKernel", false);
    gvdb_.mPool->RetrieveMem(p);
    // cuCtxSynchronize();
  }
  // Free the gpu memory
  cudaCheck(cuMemFree(p.gpu), "GVDBMRFMap", "get_alphas", "cuMemFree", "", false);
  cudaCheck(cuMemFree(cuVDBInfo), "GVDBMRFMap", "get_alphas", "cuMemFree", "", false);
  pop_ctx();
}

void GVDBMRFMap::get_occupied_voxels_mask(Eigen::Ref<RMatXi> voxel_mask, float threshold) {
  push_ctx();
  // Go brick-by-brick, filling up the mask in one go
  gvdb_.SetModule(gvdb_module_);

  // Send VDB Info (*to user module*)
  gvdb_.PrepareVDB();
  CUdeviceptr cuVDBInfo;
  cudaCheck(cuMemAlloc(&cuVDBInfo, sizeof(VDBInfo)), "GVDBMRFMap", "get_alphas", "cuMemAlloc", "cuVDBInfo", false);
  cudaCheck(cuMemcpyHtoD(cuVDBInfo, gvdb_.getVDBInfo(), sizeof(VDBInfo)), "GVDBMRFMap", "get_alphas", "cuMemcpyHtoD",
            "cuVDBInfo", false);

  // Get number of bricks
  Node *node;
  Vector3DI axisres = gvdb_.mPool->getAtlasRes(0);
  Vector3DI pos;
  int3 res = make_int3(gvdb_.getRes(0), gvdb_.getRes(0), gvdb_.getRes(0));
  int sz = gvdb_.getVoxCnt(0) / 8;  // res*res*res bits/8 bytes
  int leafcnt = gvdb_.mPool->getPoolTotalCnt(0, 0);
  if (voxel_mask.rows() != leafcnt || voxel_mask.cols() != sz / sizeof(int)) {
    throw std::runtime_error("Dimensions for the voxel mask array (" + std::to_string(voxel_mask.rows()) + "," + std::to_string(voxel_mask.cols()) + ") are incorrect! (!=" + std::to_string(leafcnt) + "," + std::to_string(sz / sizeof(int)));
    return;
  }
  DataPtr p;
  p.alloc = gvdb_.mPool;
  p.lastEle = sz;
  p.usedNum = sz;
  p.max = sz;
  p.stride = 1;
  p.size = (uint64)sz * (uint64)1;
  p.subdim = Vector3DI(0, 0, 0);
  // Allocate memory for the gpu equiv that the data will be transferred to
  cudaCheck(cuMemAlloc(&p.gpu, p.size), "GVDBMRFMap", "get_occupied_voxels_mask", "cuMemAlloc", "", false);

  for (int n = 0; n < leafcnt; n++) {
    node = gvdb_.getNode(0, 0, n);
    pos = node->mPos; 
    // Assign it 0 values
    cudaCheck(cuMemsetD32(p.gpu, 0, p.size / sizeof(int)), "GVDBMRFMap", "get_occupied_voxels_mask", "cuMemsetD32", "", false);

    // Assign the alphas memory offset
    p.cpu = reinterpret_cast<char *>(voxel_mask.data()) + n * voxel_mask.rowStride() * sizeof(int);
    int chan = ALPHAS;

    void *args[6] = {&cuVDBInfo, &node->mValue, &res, &p.gpu, &chan, &threshold};
    cudaCheck(cuLaunchKernel(get_occupied_voxel_mask_kernel_, 1, 1, 1, res.x, res.y, res.z, 0, NULL, args, NULL), "get_occupied_voxels_mask",
              "get_occupied_voxel_mask_kernel_", "cuLaunch", "cuLaunchKernel", false);
    gvdb_.mPool->RetrieveMem(p);
  }
  cuCtxSynchronize();
  // Free the gpu memory
  cudaCheck(cuMemFree(p.gpu), "GVDBMRFMap", "get_alphas", "cuMemFree", "", false);
  cudaCheck(cuMemFree(cuVDBInfo), "GVDBMRFMap", "get_alphas", "cuMemFree", "", false);
  pop_ctx();
}
#include <mrfmap/GVDBCamera.h>
#include <mrfmap/GVDBImage.h>
#include <mrfmap/GVDBMRFMap.h>

extern gvdb_params_t gvdb_params_;

void convert_pose_to_opengl_proj_mat(const Eigen::MatrixXf &pose, TMat &view_mat, TMat &proj_mat) {
  Eigen::Matrix4f opengl_in_cam;
  opengl_in_cam << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
  Eigen::Matrix4f opengl_in_world = pose * opengl_in_cam;

  // Scale metric by resolution
  opengl_in_world.col(3).head<3>() /= gvdb_params_.res;
  // Offset to center of volume due to requirements of gvdb to only have positive coordinates
  Eigen::Vector3f scaled_map_center_coords = gvdb_params_.dims * 0.5f / gvdb_params_.res;
  //Don't shift z coordinate
  scaled_map_center_coords[2] = 0.0f;
  opengl_in_world.col(3).head<3>() += scaled_map_center_coords;
  view_mat = opengl_in_world.inverse();

  // From http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
  // we have
  float mNear = 0.1f, mFar = Z_MAX / gvdb_params_.res;
  proj_mat(0, 0) = gvdb_params_.K(0, 0) / gvdb_params_.K(0, 2);
  proj_mat(1, 1) = gvdb_params_.K(1, 1) / gvdb_params_.K(1, 2);
  proj_mat(2, 2) = -(mFar + mNear) / (mFar - mNear);
  proj_mat(2, 3) = -(2.0f * mFar * mNear) / (mFar - mNear);
  proj_mat(3, 2) = -1.0f;

  view_mat.transposeInPlace();
  proj_mat.transposeInPlace();
}

void GVDBCamera::gpuAllocate() {
  DEBUG(std::cout << "Creating new camera\n");
#ifdef FULL_INFERENCE
  // If we're in full inference mode, add unique channel for weighted_incoming
  std::cout<<"Adding channel "<<ALPHAS + cam_id_ + 1<<"\n";
  map_->gvdb_.AddChannel(ALPHAS + cam_id_ + 1, T_FLOAT, 1, F_LINEAR, F_CLAMP, Vector3DI(0, 0, 0), false);
#endif
  cam_ = new Camera3D;  // Create Camera
  cam_->mNear = 0.1f;
  cam_->mFar = Z_MAX / gvdb_params_.res;
  // cam_->mAspect = cols_ / rows_;
  DEBUG(std::cout << "cam_id is " << cam_id_ << "\n"; std::cout << "pose is " << pose_ << "\n";
        std::cout << "K is " << gvdb_params_.K << "\n";);
  // Add an empty data pointer to the render buffer
  DataPtr new_buffer;
  new_buffer.alloc = map_->gvdb_.mPool;
  new_buffer.cpu = 0x0;  // no cpu residence yet
  new_buffer.usedNum = 0;
  new_buffer.lastEle = 0;
  new_buffer.garray = 0;
  new_buffer.grsc = 0;
  new_buffer.glid = -1;
  new_buffer.gpu = 0x0;
  new_buffer.stride = gvdb_params_.cols;
  new_buffer.max = gvdb_params_.cols * gvdb_params_.rows;
  map_->gvdb_.mRenderBuf.push_back(new_buffer);
#ifndef FULL_INFERENCE
  // Allocate device memory for last_outgoing index image
  cuMemAlloc(&d_last_outgoing_img_, gvdb_params_.rows * gvdb_params_.cols * sizeof(uchar));
  cuMemsetD8(d_last_outgoing_img_, 0, gvdb_params_.rows * gvdb_params_.cols);
  // And now assign this to a new corresponding render buffer entry
  map_->gvdb_.mRenderBuf.push_back(new_buffer);
  map_->gvdb_.mRenderBuf.back().gpu = d_last_outgoing_img_;
#endif
}

void GVDBCamera::reprojectAndActivate() {
  // Rotate this point into the fixed world frame using the camera pose
  GVDBImage::MRow reproj_points =
      (pose_ * img_->get_reproj_points().rowwise().homogeneous().transpose()).colwise().hnormalized().transpose();
  // Scale points and offset to fit within map dims
  reproj_points *= 1.0f / gvdb_params_.res;
  Eigen::Vector3f scaled_map_center_coords = gvdb_params_.dims * 0.5f / gvdb_params_.res;
  //Don't shift z coordinate
  scaled_map_center_coords[2] = 0.0f;
  reproj_points = reproj_points.rowwise() + scaled_map_center_coords.transpose();

  // Make sure that we respect the global map bounds to not activate space beyond
  // std::vector<int> points_to_keep;
  GVDBImage::MRow gated_reproj_points = GVDBImage::MRow::Zero(reproj_points.rows(), reproj_points.cols());
  int num_points = -1;
  for (int i = 0; i < reproj_points.rows(); ++i) {
    Eigen::Vector3f p = reproj_points.row(i);
    if ((p[0] < 0 || p[0] >= gvdb_params_.dims[0] / gvdb_params_.res) ||
        (p[1] < 0 || p[1] >= gvdb_params_.dims[1] / gvdb_params_.res) ||
        (p[2] < 0 || p[2] >= gvdb_params_.dims[2] / gvdb_params_.res)) {
      // This point is out of bounds; do nothing
    } else {
      gated_reproj_points.row(++num_points) = reproj_points.row(i);
    }
  }
  if (num_points == -1) {
    std::cout << "WARNING:: None of the reprojected points were within the volume!!!\n";
  } else {
    gated_reproj_points.conservativeResize(num_points, Eigen::NoChange);
    map_->activateSpace(gated_reproj_points);
  }
    // Also flag the cached scene to be updated within GVDB
  needs_update_ = true;

// Copy over image buffer address to this map
#ifndef FULL_INFERENCE
  map_->gvdb_.mRenderBuf[cam_id_ * 2 + 1].gpu = img_->get_buf();
#else
  map_->gvdb_.mRenderBuf[cam_id_ + 1].gpu = img_->get_buf();
#endif
  // Output memory usage
  // DEBUG(get_mem_usage(););
}

void GVDBCamera::activateCamera() {
#ifndef FULL_INFERENCE
  map_->gvdb_.getScene()->SetDepthBuf(cam_id_ * 2 + 1);
#else
  map_->gvdb_.getScene()->SetDepthBuf(cam_id_ + 1);
#endif
  TMat view_mat = TMat::Zero(4, 4), proj_mat = TMat::Zero(4, 4);
  convert_pose_to_opengl_proj_mat(pose_, view_mat, proj_mat);

  cam_->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
  cam_->updateFrustum();
  map_->gvdb_.getScene()->mCamera = cam_;
  // Since we're caching scenes, let's activate this camera and add to cache
  if (!is_cached_) {
    map_->gvdb_.CacheScene();
    is_cached_ = true;
    needs_update_ = false;
  } else if(needs_update_ ){
    map_->gvdb_.CacheScene(cam_id_);
    needs_update_ = false;
  } else{
    // Nothing to be done, we will just used the cached scene at time of render.
  }
}

void GVDBCamera::calculate_cum_len() {
  cum_len_calculated_ = true;
}